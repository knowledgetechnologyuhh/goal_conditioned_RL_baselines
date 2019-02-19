(define (problem pb1) (:domain tower)
 (:init
	 (not (o0_on_o2))
	 (gripper_at_o0)
	 (not (gripper_at_o2))
	 (not (o2_on_o1))
	 (not (o2_on_o0))
	 (o0_on_o1)
	 (not (o1_on_o2))
	 (gripper_at_o1)
	 (not (o1_at_target))
	 (not (o0_at_target))
	 (not (o1_on_o0))
	 (not (gripper_at_target))
	 (not (o2_at_target))
)
(:goal (and
	 (o0_at_target)
	 (o1_on_o0)
	 (o2_on_o1)

))


)
