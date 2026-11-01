# 检查12组正式实验的覆盖率、输入哈希及保存的答案。
import gzip
import hashlib
import json
from pathlib import Path

from searchlab.generators import load_tree
from searchlab.grid import Grid
from searchlab.reference import check_route, reference_tree, shortest_distances
from searchlab.grid import SearchResult

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'Q1-01':123,'Q1-02':63,'Q1-03':12,'Q1-04':366,
          'Q2-01':15,'Q2-02':3,'Q2-03':86,'Q2-04':360,
          'Q3-01':103,'Q3-02':480,'Q3-03':720,'Q3-04':320}


def main():
    all_rows=[];instances={};seen=set();counts={}
    for ident,count in EXPECTED.items():
        path=ROOT/'outputs/raw'/f'{ident}.jsonl'
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==count,(ident,len(rows),count)
        counts[ident]=len(rows)
        for row in rows:
            assert row['run_key'] not in seen,row['run_key']
            seen.add(row['run_key'])
            assert row['timing_mode']=='full'
            assert row['status'] in ('ok','unreachable'),row['status']
            assert all(t>0 for t in row['timings_ns'])
            digest=row['instance_sha256']
            if digest not in instances:
                raw=gzip.decompress((ROOT/'outputs/instances'/(digest+'.json.gz')).read_bytes())
                assert hashlib.sha256(raw).hexdigest()==digest
                obj=json.loads(raw)
                if obj['kind']=='tree':
                    obj={'kind':'tree','_reference':reference_tree(load_tree(obj))}
                instances[digest]=obj
            obj=instances[digest]
            if obj['kind']=='tree':
                assert row['value']==row['expected_value']==obj['_reference']
            elif obj['kind']=='grid':
                g=Grid(obj['grid'],obj['start'],obj['goal'],obj['problem'])
                ref=shortest_distances(g.costs,g.start).get(g.goal,-1)
                # 题目二的普通曼哈顿距离用于反例诊断；广度优先搜索求最少步数。
                if row['algorithm']=='A*' and not (obj['problem']=='weighted' and row['heuristic']=='manhattan'):
                    res=SearchResult(path=[tuple(n) for n in row['path']],cost=row['cost'],status=row['status'])
                    check_route(g,res,ref)
            elif obj['kind']=='multiquery_grid':
                g=Grid(obj['grid'],obj['starts'][0],obj['goal'],'weighted')
                ref=shortest_distances(g.costs,g.goal,reverse=True)
                assert row['query_costs']==[ref[tuple(n)] for n in row['query_starts']]
            all_rows.append(row)
    report={'status':'passed','experiment_groups':len(counts),'run_rows':len(all_rows),
            'unique_instances':len(instances),'counts':counts,'input_hashes_verified':True,
            'answers_cross_checked':True,'timing_mode':'full'}
    (ROOT/'outputs/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
