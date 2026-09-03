# perception이 발행하고 grasp만 구독하는 내부용 토픽. WorldState와 stamp로 동기화해서 매칭한다.
# grasp는 이 마스크로 포인트클라우드를 자기 프로세스 안에서만 재구성하며, 그 결과를 다시 밖으로 내보내지 않는다.

string schema_version
string trace_id
builtin_interfaces/Time stamp
string[] object_ids            # WorldState.objects의 object_id와 1:1 대응
sensor_msgs/Image[] masks      # object_id별 단일 채널 마스크 (mono8, 0/255)
