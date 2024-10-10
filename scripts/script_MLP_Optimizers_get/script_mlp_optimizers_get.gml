#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	LIST OF AVAILABLE GRADIENT DESCENT OPTIMIZERS
	
	Used to get index for optimizer, which then can be send to extension.
____________________________________________________________________________________________________
*/
#endregion

enum OptimizerType {
	UNDEFINED, STOCHASTIC, MOMENTUM, NESTEROV, ADAM, ADAGRAD, ADADELTA, ADAMAX, NADAM, RMSPROP, AMSGRAD, length
}

#macro	___OPTIMIZERS global.___gGRADIENT_DESCENT_OPTIMIZERS_ARRAY
___OPTIMIZERS = []
___OPTIMIZERS[OptimizerType.STOCHASTIC] = Stochastic;
___OPTIMIZERS[OptimizerType.MOMENTUM]	= Momentum;
___OPTIMIZERS[OptimizerType.NESTEROV]	= Nesterov;
___OPTIMIZERS[OptimizerType.ADAM]		= Adam;
___OPTIMIZERS[OptimizerType.ADAGRAD]	= AdaGrad;
___OPTIMIZERS[OptimizerType.ADADELTA]	= AdaDelta;
___OPTIMIZERS[OptimizerType.ADAMAX]		= AdaMax;
___OPTIMIZERS[OptimizerType.NADAM]		= Nadam;
___OPTIMIZERS[OptimizerType.RMSPROP]	= RMSprop;
___OPTIMIZERS[OptimizerType.AMSGRAD]	= AMSgrad;

/*____________________________________________________________________________________________________
*/

/// @func	gradient_descent_optimizer_enum(optimizer);
/// @desc	Returns enum-index from given optimizer.
function gradient_descent_optimizer_enum(funktion) {
	switch(funktion) {	
		case Stochastic:	return OptimizerType.STOCHASTIC; 
		case Momentum:		return OptimizerType.MOMENTUM;	
		case Nesterov:		return OptimizerType.NESTEROV;
		case Adam:			return OptimizerType.ADAM;
		case AdaGrad:		return OptimizerType.ADAGRAD;
		case AdaDelta:		return OptimizerType.ADADELTA;
		case AdaMax:		return OptimizerType.ADAMAX;
		case Nadam:			return OptimizerType.NADAM;
		case RMSprop:		return OptimizerType.RMSPROP;
		case AMSgrad:		return OptimizerType.AMSGRAD;
		default: 
			PRINT("No known enum for given function. Returned index for 'undefined'."); 
			return OptimizerType.UNDEFINED;
	}
}

/*____________________________________________________________________________________________________
*/

/// @func	gradient_descent_optimizer_list();
/// @desc	Returns string-list of available gradient-descent optimizers. 
function gradient_descent_optimizer_list() {
	var i, text = "| ";
	for(i = 0; i < array_length(___OPTIMIZERS); i++) {
		text +=  string(i) + " : " + script_get_name(___OPTIMIZERS[i]) + " | ";
	}	return text;
}

/*____________________________________________________________________________________________________
*/