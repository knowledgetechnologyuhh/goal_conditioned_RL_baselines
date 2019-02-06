(define (problem pb1) (:domain dinner)
 (:init
	 (o1_on_o0)
	 (not (gripper_at_o0))
	 (not (grasped_o0))
	 (not (gripper_open))
	 (not (o0_on_o1))
	 (not (grasped_o1))
	 (not (o0_at_target))
	 (not (gripper_at_o1))
	 (not (o1_at_target))
	 (not (gripper_at_target))
)
(:goal (and 
	 (o0_at_target)
	 (gripper_at_target)
))


)
