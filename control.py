#!/usr/bin/env python
# license removed for brevity
import rospy
from ackermann_msgs.msg import AckermannDrive 

def talker():
    angle_new = 0.5
    angle_old = 0.4
    acceleration = 0.5
    velocity = 0
    deltat = 0.1
    pub = rospy.Publisher('/carla/ego_vehicle/ackermann_cmd', AckermannDrive, queue_size=10)
    rospy.init_node('CAV_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    angle_vel = 2
    speed = 10
    jerk = 1
    while not rospy.is_shutdown():
        # control_cmd = "{steering_angle: %f, steering_angle_velocity: %f, speed: %f, acceleration: %f, jerk: 0.0}" % (angle_new,angle_vel,speed,acceleration)
        #control_cmd = ['%f'%(angle_new),'%f'%(angle_vel),'%f'%(speed),'%f'%(acceleration),'%f'%(jerk)]
        control_cmd = AckermannDrive()
        control_cmd.steering_angle = angle_new
        control_cmd.steering_angle_velocity = angle_vel
        control_cmd.speed = speed
        control_cmd.acceleration = acceleration
        control_cmd.jerk = jerk
        rospy.loginfo(control_cmd)
        pub.publish(control_cmd)
        rate.sleep()
 
if __name__ == '__main__':
    try:
        talker()
        print('Publish start')
    except rospy.ROSInterruptException:
        pass
