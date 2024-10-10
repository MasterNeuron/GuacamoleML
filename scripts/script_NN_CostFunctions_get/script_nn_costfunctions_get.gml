#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________________________________________

	List of Cost functions for neural network optimizers
____________________________________________________________________________________________________________________________________
*/
#endregion


enum CostFunc {
	DELTA, 
	MEANSQUARE, MEANABSOLUTE, MEANSQUAREDLOG, ROOTMEANSQUARE, ROOTMEANSQUARELOG, 
	HUBER, HINGE, SMOOTHHINGE, QUADHINGE, LOGCOSH,
	CROSSENTROPY, CATEGORIAL_CROSSENTROPY, BINARY_CROSSENTROPY, MULTILABEL_CROSSENTROPY,
	EXPONENTIAL, HELLINGER, KULLBACKLEIBLER, GENERALIZED_KL, ITAKURASAITO, 
	length
}

#macro	___COSTFUNCTIONS global.___gCOST_FUNCTION_ARRAY
___COSTFUNCTIONS = [];
___COSTFUNCTIONS[CostFunc.DELTA]					= Delta;
___COSTFUNCTIONS[CostFunc.MEANSQUARE]				= MeanSquare;
___COSTFUNCTIONS[CostFunc.MEANABSOLUTE]				= MeanAbsolute;
___COSTFUNCTIONS[CostFunc.MEANSQUAREDLOG]			= MeanSquareLog;
___COSTFUNCTIONS[CostFunc.ROOTMEANSQUARE]			= RootMeanSquare;
___COSTFUNCTIONS[CostFunc.ROOTMEANSQUARELOG]		= RootMeanSquareLog;
___COSTFUNCTIONS[CostFunc.HUBER]					= Huber;
___COSTFUNCTIONS[CostFunc.HINGE]					= Hinge;
___COSTFUNCTIONS[CostFunc.SMOOTHHINGE]				= SmoothHinge;
___COSTFUNCTIONS[CostFunc.QUADHINGE]				= QuadHinge;
___COSTFUNCTIONS[CostFunc.LOGCOSH]					= LogCosh;
___COSTFUNCTIONS[CostFunc.CROSSENTROPY]				= CrossEntropy;
___COSTFUNCTIONS[CostFunc.CATEGORIAL_CROSSENTROPY]	= CategorialCE;
___COSTFUNCTIONS[CostFunc.BINARY_CROSSENTROPY]		= BinaryCE;
___COSTFUNCTIONS[CostFunc.MULTILABEL_CROSSENTROPY]	= MultiCE;
___COSTFUNCTIONS[CostFunc.EXPONENTIAL]				= Exponential;
___COSTFUNCTIONS[CostFunc.HELLINGER]				= Hellinger;
___COSTFUNCTIONS[CostFunc.KULLBACKLEIBLER]			= KullbackLeibler;
___COSTFUNCTIONS[CostFunc.GENERALIZED_KL]			= GeneralizedKL;
___COSTFUNCTIONS[CostFunc.ITAKURASAITO]				= ItakuraSaito;

// Making these as Method. Executes faster. 
//  -> Otherwise would always create temporal method when calling.
for(var i = 0; i < CostFunc.length; i++) {
	___COSTFUNCTIONS[i] = method(undefined, ___COSTFUNCTIONS[i]);
}	// Though this makes you cannot get name simply by "script_get_name".


/*____________________________________________________________________________________________________________________________________
*/

/// @func	cost_function_enum(function);
/// @desc	Returns enum-index from given function.
function cost_function_enum(funktion) {
	switch(funktion) {	
		case Delta:				return CostFunc.DELTA;
		case MeanSquare:		return CostFunc.MEANSQUARE;
		case MeanAbsolute:		return CostFunc.MEANABSOLUTE;
		case MeanSquareLog:		return CostFunc.MEANSQUAREDLOG;
		case RootMeanSquare:	return CostFunc.ROOTMEANSQUARE;
		case RootMeanSquareLog:	return CostFunc.ROOTMEANSQUARELOG;
		case Huber:				return CostFunc.HUBER;
		case Hinge:				return CostFunc.HINGE;
		case SmoothHinge:		return CostFunc.SMOOTHHINGE;
		case QuadHinge:			return CostFunc.QUADHINGE;
		case LogCosh:			return CostFunc.LOGCOSH;
		case CrossEntropy:		return CostFunc.CROSSENTROPY;
		case CategorialCE:		return CostFunc.CATEGORIAL_CROSSENTROPY;
		case BinaryCE:			return CostFunc.BINARY_CROSSENTROPY;
		case MultiCE:			return CostFunc.MULTILABEL_CROSSENTROPY;
		case Exponential:		return CostFunc.EXPONENTIAL;	
		case Hellinger:			return CostFunc.HELLINGER;
		case KullbackLeibler:	return CostFunc.KULLBACKLEIBLER;
		case GeneralizedKL:		return CostFunc.GENERALIZED_KL;
		case ItakuraSaito:		return CostFunc.ITAKURASAITO;
		default: 
			PRINT("No known enum for given function. Returned index for 'Delta'."); 
			return CostFunc.DELTA;
	}
}


/*____________________________________________________________________________________________________________________________________
*/

/// @func	cost_function_list();
/// @desc	Returns string-list of available gradient-descent optimizers. 
function cost_function_list() {
	var i, text = "| ";
	for(i = 0; i < array_length(___COSTFUNCTIONS); i++) {
		text +=  string(i) + " : " + script_get_name(___COSTFUNCTIONS[i]) + " | ";
	}	return text;
}

/*____________________________________________________________________________________________________________________________________
*/

/// @func	cost_function_name(enumIndex);
/// @desc	Returns activation function as string
function cost_function_name(enumIndex) {
	switch(enumIndex) {
		case CostFunc.DELTA:					return script_get_name(Delta);			
		case CostFunc.MEANSQUARE:				return script_get_name(MeanSquare);	
		case CostFunc.MEANABSOLUTE:				return script_get_name(MeanAbsolute);		
		case CostFunc.MEANSQUAREDLOG:			return script_get_name(MeanSquareLog);		
		case CostFunc.ROOTMEANSQUARE:			return script_get_name(RootMeanSquare);	
		case CostFunc.ROOTMEANSQUARELOG:		return script_get_name(RootMeanSquareLog);	
		case CostFunc.HUBER:					return script_get_name(Huber);				
		case CostFunc.HINGE:					return script_get_name(Hinge);				
		case CostFunc.SMOOTHHINGE:				return script_get_name(SmoothHinge);	
		case CostFunc.QUADHINGE:				return script_get_name(QuadHinge);			
		case CostFunc.LOGCOSH:					return script_get_name(LogCosh);			
		case CostFunc.CROSSENTROPY:				return script_get_name(CrossEntropy);		
		case CostFunc.CATEGORIAL_CROSSENTROPY:	return script_get_name(CategorialCE);		
		case CostFunc.BINARY_CROSSENTROPY:		return script_get_name(BinaryCE);			
		case CostFunc.MULTILABEL_CROSSENTROPY:	return script_get_name(MultiCE);		
		case CostFunc.EXPONENTIAL:				return script_get_name(Exponential);	
		case CostFunc.HELLINGER:				return script_get_name(Hellinger);			
		case CostFunc.KULLBACKLEIBLER:			return script_get_name(KullbackLeibler);
		case CostFunc.GENERALIZED_KL:			return script_get_name(GeneralizedKL);		
		case CostFunc.ITAKURASAITO:				return script_get_name(ItakuraSaito);		
		default: return "Unknown cost function.";
	}									
}

/*___________________________________________________________________________________________________________________________________
*/

