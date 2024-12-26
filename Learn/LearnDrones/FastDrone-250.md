# Fast-Drone 250

[eigen error: no match for ‘operator=’](https://github.com/HKUST-Aerial-Robotics/Fast-Planner/issues/92)

Important:

- lower eigen from 3.4.0 to 3.3.7
- lower liblog4cxx from 1.0.0 to 0.10.0

dependent ros package (noetic):

- [control_toolbox - melodic-devel](https://github.com/ros-controls/control_toolbox/tree/melodic-devel)
- [ddynamic_reconfigure - kinetic-devel](https://github.com/pal-robotics/ddynamic_reconfigure/tree/kinetic-devel)
- [geographic_info - master](https://github.com/ros-geographic-info/geographic_info/tree/master)
- [geometry2 - noetic-devel](https://github.com/ros/geometry2/tree/noetic-devel)
- [mavlink-gbp-release - release/noetic/mavlink](https://github.com/mavlink/mavlink-gbp-release/tree/release/noetic/mavlink)
- [mavros - master](https://github.com/mavlink/mavros/tree/master)
- [pcl_msgs - noetic-devel](https://github.com/ros-perception/pcl_msgs/tree/noetic-devel)
- [perception_pcl - melodic-devel](https://github.com/ros-perception/perception_pcl/tree/melodic-devel)
- [realtime_tools - melodic-devel](https://github.com/ros-controls/realtime_tools/tree/melodic-devel)
- [unique_identifier - master](https://github.com/ros-geographic-info/unique_identifier/tree/master)

in your catkin workspace:

```bash
catkin_make_isolated --install \
-DCMAKE_BUILD_TYPE=Release \
-DPYTHON_EXECUTABLE=/usr/bin/python3 
```