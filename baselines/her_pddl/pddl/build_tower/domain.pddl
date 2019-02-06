(define (domain dinner) (:requirements :strips) 
(:predicates 
	grasped_o0
	grasped_o1
	gripper_at_o0
	gripper_at_o1
	gripper_at_target
	gripper_open
	o0_at_target
	o0_on_o1
	o1_at_target
	o1_on_o0
)
(:action open_gripper 
	:parameters () 
	:precondition () 
	:effect (and (gripper_open) (not (grasped_o0))(not (grasped_o1)) )
)

(:action grasp_o0 
	:parameters () 
	:precondition (and (gripper_at_o0)  (not (o1_on_o0))) 
	:effect (and (grasped_o0) (not (gripper_open)))
)

(:action move_gripper_to_o0 
	:parameters () 
	:precondition (gripper_open) 
	:effect (and (gripper_at_o0) (not (gripper_at_target)) (not (gripper_at_o1)))
)

(:action move_o0_to_target 
	:parameters () 
	:precondition (and (grasped_o0)  (not (o1_on_o0))) 
	:effect (o0_at_target)
)

(:action move_o0_on_o1 
	:parameters () 
	:precondition (and (grasped_o0)  (not (o0_on_o1))) 
	:effect (o0_on_o1)
)

(:action grasp_o1 
	:parameters () 
	:precondition (and (gripper_at_o1)  (not (o0_on_o1))) 
	:effect (and (grasped_o1) (not (gripper_open)))
)

(:action move_gripper_to_o1 
	:parameters () 
	:precondition (gripper_open) 
	:effect (and (gripper_at_o1) (not (gripper_at_target)) (not (gripper_at_o0)))
)

(:action move_o1_to_target 
	:parameters () 
	:precondition (and (grasped_o1)  (not (o0_on_o1))) 
	:effect (o1_at_target)
)

(:action move_gripper_to_target 
	:parameters () 
	:precondition (gripper_open) 
	:effect (and (gripper_at_target) (not (gripper_at_o0))(not (gripper_at_o1)))
)

)
