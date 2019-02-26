(define (domain tower) (:requirements :strips)
(:predicates
	gripper_at_o0
	gripper_at_o1
	gripper_at_o2
	gripper_at_target
	o0_at_target
	o0_on_o1
	o0_on_o2
	o1_at_target
	o1_on_o0
	o1_on_o2
	o2_at_target
	o2_on_o0
	o2_on_o1
)
(:action move_gripper_to__o0
	:parameters ()
	:precondition ()
	:effect (and (gripper_at_o0) (not (gripper_at_o1))(not (gripper_at_o2)) (not (gripper_at_target)) )
)

(:action move__o0_to_target
	:parameters ()
	:precondition (and (gripper_at_o0) )
	:effect (and (o0_at_target) )
)

(:action move_gripper_to__o1
	:parameters ()
	:precondition ()
	:effect (and (gripper_at_o1) (not (gripper_at_o0))(not (gripper_at_o2)) (not (gripper_at_target)) )
)

(:action move__o1_to_target
	:parameters ()
	:precondition (and (gripper_at_o1) )
	:effect (and (o1_at_target) )
)

(:action move__o1_on__o0
	:parameters ()
	:precondition (and (gripper_at_o1)   (not (o1_on_o0)) (not (o2_on_o0)))
	:effect (and (o1_on_o0) )
)

(:action move_gripper_to__o2
	:parameters ()
	:precondition ()
	:effect (and (gripper_at_o2) (not (gripper_at_o0))(not (gripper_at_o1)) (not (gripper_at_target)) )
)

(:action move__o2_to_target
	:parameters ()
	:precondition (and (gripper_at_o2) )
	:effect (and (o2_at_target) )
)

(:action move__o2_on__o1
	:parameters ()
	:precondition (and (gripper_at_o2)   (not (o0_on_o1)) (not (o2_on_o1)))
	:effect (and (o2_on_o1) )
)

(:action move_gripper_to_target
	:parameters ()
	:precondition (and (not (grasped_o0))(not (grasped_o1))(not (grasped_o2)))
	:effect (and (gripper_at_target) (not (gripper_at_o0))(not (gripper_at_o1))(not (gripper_at_o2)))
)

)
