from run_stage7 import phase_wrapper

def test():
    events=[]
    def sync():events.append('sync')
    def proc(x,*,power):events.append(('proc',x,power));return {'result':x**power}
    wrap=phase_wrapper(proc,'dino',events_append:=[],sync)
    assert wrap(3,power=2)=={'result':9}
    assert events[0]=='sync' and events[-1]=='sync'
    assert len(events_append)==1 and events_append[0]['stage']=='dino' and events_append[0]['ms']>=0
    assert len(events)==3
    print('OFFLINE_PROFILER_TEST_PASS')
if __name__=='__main__':test()
