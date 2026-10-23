"""Build all-user CKE and timestamp-filtered evidence for paper-order inputs.

Only historical events observed by the last context check-in are eligible.
Preserve upstream user order, undirected adjacency order, last-row attributes,
context exclusion and its effective eleven-neighbor limit. Do not alter upstream.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ['inputs', 'upstream', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if not __debug__:
        raise RuntimeError('Run without -O')
    assert not a.output.exists(), 'Never overwrite a previous graph attempt'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    import numpy as np
    import pandas as pd
    import networkx as nx
    import glob
    import itertools
    started = time.time()
    assert (pd.__version__, np.__version__, nx.__version__) == ('2.1.4', '1.26.4', '3.3')
    r = json.loads((a.inputs / 'result.json').read_text())
    assert r['status'] == 'paper_order_selected_inputs_pass'
    assert sha(a.inputs / 'result.json') == '4bedb61d371a670701454b8b23494a986a82acbbf3a8fa78ab6c9d94cbe9e81f'
    validation = json.loads((a.inputs / 'validation.json').read_text())
    assert validation['status'] == 'serialized_paper_order_inputs_reference_pass'
    assert validation['result_sha256'] == sha(a.inputs / 'result.json')
    for name, meta in r['artifacts'].items():
        assert sha(a.inputs / name) == meta['sha256'], name
    world = a.upstream / 'models/world_model.py'
    assert sha(world) == '7b251f5237b7b87fa70f7061ec26c1a88031798752048210c8dc6f7803d71f5c'
    cls = next(n for n in ast.parse(world.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SocialWorld')
    ns = dict(nx=nx, pd=pd, os=os, glob=glob, itertools=itertools, Counter=Counter)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(world), 'exec'), ns)
    Social = ns['SocialWorld']
    data = pd.read_pickle(a.inputs / 'filtered_session_rows.pkl')
    manifest = pd.read_csv(a.inputs / 'session_manifest.csv')
    tests = json.loads((a.inputs / 'test_dictionary_Tokyo_paper_order.json').read_text())
    truths = json.loads((a.inputs / 'true_locations_Tokyo_paper_order.json').read_text())
    selection = json.loads((a.inputs / 'query_selection.json').read_text())
    groups = {str(u): frame for u, frame in data.groupby('user_id', sort=True)}
    assert len(groups) == 7360 and len(data) == 687542
    pools, edges, offset, full_city_proxy = [], [], 0, {}
    cared = ['hour', 'weekday', 'venue_category_name', 'venue_id', 'admin', 'subdistrict', 'poi', 'street']
    history_ids = set(manifest.loc[manifest.llm_history, 'DL_traj_id'])
    for uid, frame in groups.items():
        # All retained users, not merely the selected200 prediction dictionary.
        history = frame[frame.DL_traj_id.isin(history_ids)].copy()
        ids = frame.traj_id.unique()
        reference = frame[frame.traj_id.isin(ids[:int(.8*len(ids))])]
        assert history.index.tolist() == reference.index.tolist()
        values = history[cared].values.tolist()
        full_city_proxy[uid] = {'history_proxy': {'historical_stays_long': values}}
        if uid in tests:
            assert values == next(iter(tests[uid].values()))['historical_stays_long']
        history['stamp'] = history.datetime.astype('int64')
        history['ordinal'] = np.arange(offset, offset + len(history))
        offset += len(history)
        h = list(zip(history.venue_id, history.stamp))
        edges.extend((s[0], t[0], max(s[1], t[1])) for s, t in zip(h[:-1], h[1:]))
        pools.append(history)
    pool = pd.concat(pools, ignore_index=True)
    assert len(pool) == r['llm_history_records'] == 556513
    edge = pd.DataFrame(edges, columns=['src', 'dst', 'stamp'])
    a.output.mkdir()
    (a.output / 'status.json').write_text(json.dumps(dict(stage='building_full_city_graph', pid=os.getpid(), started_at=started)))
    proxy_dataset = SimpleNamespace(get_generated_datasets=lambda: (full_city_proxy, {}))
    social = Social(proxy_dataset, str(a.output), 'Tokyo', khop=1, max_neighbors=10)
    # Recompute weighted edge map independently and verify serialized roundtrip.
    counts = Counter((s,t) for s,t,stamp in edges)
    reference_graph = nx.Graph()
    reference_graph.add_edges_from((s,t,{'weight':w}) for (s,t),w in counts.items())
    assert set(reference_graph) == set(social.graph)
    assert {frozenset(e) for e in reference_graph.edges()} == {frozenset(e) for e in social.graph.edges()}
    assert all(reference_graph[s][t] == attrs for s,t,attrs in social.graph.edges(data=True))
    last_rows = pool.drop_duplicates('venue_id',keep='last').set_index('venue_id')
    for node,attrs in social.graph.nodes(data=True):
        row = last_rows.loc[node]
        assert attrs == dict(category=row.venue_category_name, admin=row.admin, subdistrict=row.subdistrict, street=row.street, poi=row.poi)
    reloaded = nx.read_gml(a.output / 'Tokyo_graph.gml')
    assert dict(reloaded.nodes(data=True)) == dict(social.graph.nodes(data=True))
    assert all(reloaded[s][t] == attrs for s,t,attrs in social.graph.edges(data=True))
    assert reloaded.number_of_edges() == social.graph.number_of_edges()
    social.graph = reloaded
    del full_city_proxy, proxy_dataset, reference_graph, last_rows
    node_groups = {v: frame for v, frame in pool.groupby('venue_id', sort=False)}
    incident = {}
    for i, (s, t, stamp) in enumerate(edges):
        incident.setdefault(s, []).append((t, stamp, i))
        if s != t:
            incident.setdefault(t, []).append((s, stamp, i))
    bundle, audits = [], []
    for qi, (uid, tid) in enumerate(selection):
        ex = tests[uid][tid]
        session = groups[uid][groups[uid].traj_id == int(tid)]
        cutoff = int(session.iloc[-2].datetime.value)
        target = int(session.iloc[-1].datetime.value)
        last = ex['context_stays'][-1][3]
        context = [x[3] for x in ex['context_stays']]
        assert cutoff <= target
        original_neighbors = [n for n, _ in social.retrival_neighbors(last, context)][:11]
        original_audit = []
        for n in original_neighbors:
            stamps = [stamp for neighbor, stamp, _ in incident.get(last, []) if neighbor == n]
            source = node_groups[n].iloc[-1]
            original_audit.append(dict(edge_only_after_cutoff=min(stamps) > cutoff,
                                       edge_has_later_contribution=max(stamps) > cutoff,
                                       attribute_source_after_cutoff=int(source.stamp) > cutoff))
        def reloaded_order(boundary):
            valid_edges = edge[edge.stamp <= boundary]
            # write_gml emits edges by graph node insertion order. read_gml can
            # consequently reorder adjacency; match the baseline's reload path.
            node_order = dict(zip(pd.unique(valid_edges[['src', 'dst']].values.ravel()),
                                  range(2 * len(valid_edges))))
            neighbors = list(dict.fromkeys(n for n, stamp, _ in incident.get(last, []) if stamp <= boundary))
            rank = node_order.get(last, -1)
            earlier = sorted((n for n in neighbors if node_order[n] < rank), key=node_order.get)
            return earlier + [n for n in neighbors if node_order[n] >= rank]
        unfiltered = [n for n in reloaded_order(np.iinfo(np.int64).max) if n not in context][:11]
        assert unfiltered == original_neighbors
        ordered = reloaded_order(cutoff)
        chosen = [n for n in ordered if n not in context][:11]
        strings, provenance = [], []
        for n in chosen:
            source = node_groups[n][node_groups[n].stamp <= cutoff].iloc[-1]
            strings.append(','.join([source.street, source.poi]))
            edge_time = min(stamp for neighbor, stamp, _ in incident[last] if neighbor == n and stamp <= cutoff)
            assert edge_time <= cutoff and int(source.stamp) <= cutoff
            provenance.append(dict(venue=n, edge_observed_ns=int(edge_time), attribute_source_ns=int(source.stamp)))
        info = '1-hop neighbor places in the social world:\n ' + '\n'.join(strings) if strings else ''
        # Independently construct full filtered edge graph on three fixed queries,
        # then execute unchanged upstream retrieval/formatting to check the shortcut.
        if qi in [0, 99, 199]:
            valid = edge[edge.stamp <= cutoff]
            counts = Counter(zip(valid.src, valid.dst))
            reference = Social.__new__(Social)
            reference.graph = nx.Graph()
            reference.graph.add_edges_from((s, t, {'weight': count}) for (s, t), count in counts.items())
            reference.graph = nx.parse_gml(nx.generate_gml(reference.graph))
            reference.khop, reference.max_neighbors = 1, 10
            for n in reference.graph.neighbors(last) if last in reference.graph else []:
                source = node_groups[n][node_groups[n].stamp <= cutoff].iloc[-1]
                reference.graph.nodes[n].update(category=source.venue_category_name, street=source.street, poi=source.poi)
            assert reference.get_world_info(last, context, 'address') == info
        bundle.append(dict(user=uid, trajectory=tid, inputs=ex, label=truths[uid][tid],
                           social_world_info=info, cutoff_ns=cutoff, evidence=provenance))
        audits.append(dict(query_index=qi, original_returned=len(original_neighbors), causal_returned=len(chosen),
                           original_edge_only_future=sum(x['edge_only_after_cutoff'] for x in original_audit),
                           original_edge_later_contribution=sum(x['edge_has_later_contribution'] for x in original_audit),
                           original_attribute_source_future=sum(x['attribute_source_after_cutoff'] for x in original_audit),
                           social_text_changed=info != social.get_world_info(last, context, 'address')))
        (a.output / 'status.json').write_text(json.dumps(dict(stage='constructing', pid=os.getpid(), completed=qi+1, total=200)))
    bp = a.output / 'causal_bundle.json'
    bp.write_text(json.dumps(bundle, ensure_ascii=False))
    summary = dict(status='paper_order_graph_and_causal_bundle_pass', model_calls=0, queries=200, selected_users=200,
                   cutoff_rule='last observed context check-in; event timestamps <= cutoff',
                   fixed_offline_membership=True, map_historical_validity_not_established=True,
                   retained_upstream_neighbor_limit=11, independent_graph_checks=[0, 99, 199],
                   graph_reload_order_matched=True, unfiltered_retrieval_matches_official_all_queries=True,
                   bundle_sha256=sha(bp), query_selection_sha256=r['artifacts']['query_selection.json']['sha256'],
                   input_result_sha256=sha(a.inputs / 'result.json'),
                   collective_users=len(groups), collective_history_records=len(pool),
                   graph_nodes=social.graph.number_of_nodes(), graph_edges=social.graph.number_of_edges(),
                   graph_sha256=sha(a.output / 'Tokyo_graph.gml'),
                   full_city_proxy_used=True, weighted_edges_and_node_attributes_checked=True,
                   queries_with_future_only_original_edge=sum(x['original_edge_only_future'] > 0 for x in audits),
                   queries_with_later_original_edge_contribution=sum(x['original_edge_later_contribution'] > 0 for x in audits),
                   queries_with_future_original_attribute_source=sum(x['original_attribute_source_future'] > 0 for x in audits),
                   changed_social_text_queries=sum(x['social_text_changed'] for x in audits),
                   total_causal_neighbors=sum(x['causal_returned'] for x in audits),
                   event_timestamp_violations=0, elapsed_seconds=time.time()-started,
                   script_sha256=sha(Path(__file__)), upstream_world_sha256=sha(world),
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   limits=dict(cpu_cores=2, address_space_gib=24, cpu_seconds=1800),
                   limitation='Filtered evidence condition, not verbatim official graph. Fixed benchmark membership and modern address metadata retained; not a fully historical map simulation.')
    (a.output / 'query_audit.json').write_text(json.dumps(audits, indent=2))
    (a.output / 'result.json').write_text(json.dumps(summary, indent=2))
    (a.output / 'status.json').write_text(json.dumps(dict(stage='completed', completed=200)))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
