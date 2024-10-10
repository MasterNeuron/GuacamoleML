#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________________________________________

	Activation function tables. 
	Neural networks store index which function to use. This is done to find derivative quickly
	Storing just enum-index makes easier to save/load to buffer too, as function name isn't needed.

	As design choice, activation functions and their derivatives are separate functions.
	Activation functions could include their derivative, and have argument to check wether use derivative or not. 
	-> But that makes it slower, derivative-check would always be made. 
	-> Derivatives are only needed when training in optimizer
	Also I don't know how to make automatic differentation, so I am stuck with this solution.

	If you add more functions to the list, add them at the end of it! 
	Don't change the order, otherwise Plus-mlp's wont't work correctly, as it has own function list which is similiar to these.
____________________________________________________________________________________________________________________________________
*/
#endregion

/*___________________________________________________________________________________________________________________________________
*/
enum ActFunc { 
	IDENTITY, 
	TANH, TANHLECUN, TANHSHRINK, ARCTAN, SIN, COS, 
	SIGMOID, BISIGMOID, LOGSIGMOID, LOGLOG,
	RELU, RELUCAP, LEAKYRELU, PRELU, ELU, GELU, SELU, CELU, ELISH, SWISH, 
	SOFTSIGN, SOFTPLUS, SOFTCLIP, SOFTEXP, 
	SINC, GAUSSIAN, SQRBF, ISRU, ISRLU, SQNL, 
	BENT, BINARY, THRESHOLD, ABSOLUTE,
	HARDTANH, HARDSIGMOID, HARDSHRINK, HARDSWISH, HARDELISH, 
	length
}
#macro	___ACTIVATION	global.___gACTIVATION_FUNCTIONS_ARRAY
#macro	___DERIVATIVE	global.___gACTIVATION_DERIVATIVES_ARRAY

// Activation functions									// Derivatives of Activation functions
___ACTIVATION = [];										___DERIVATIVE = [];
___ACTIVATION[ActFunc.IDENTITY]		= Identity;			___DERIVATIVE[ActFunc.IDENTITY]		= IdentityDerivative;
___ACTIVATION[ActFunc.TANH]			= Tanh;				___DERIVATIVE[ActFunc.TANH]			= TanhDerivative;
___ACTIVATION[ActFunc.TANHLECUN]	= TanhLecun;		___DERIVATIVE[ActFunc.TANHLECUN]	= TanhLecunDerivative;
___ACTIVATION[ActFunc.TANHSHRINK]	= TanhShrink;		___DERIVATIVE[ActFunc.TANHSHRINK]	= TanhShrinkDerivative;
___ACTIVATION[ActFunc.ARCTAN]		= ArcTan;			___DERIVATIVE[ActFunc.ARCTAN]		= ArcTanDerivative;
___ACTIVATION[ActFunc.SIN]			= sin;				___DERIVATIVE[ActFunc.SIN]			= cos;
___ACTIVATION[ActFunc.COS]			= cos;				___DERIVATIVE[ActFunc.COS]			= CosDerivative;
___ACTIVATION[ActFunc.SIGMOID]		= Sigmoid;			___DERIVATIVE[ActFunc.SIGMOID]		= SigmoidDerivative;
___ACTIVATION[ActFunc.BISIGMOID]	= BipolarSigmoid;	___DERIVATIVE[ActFunc.BISIGMOID]	= BipolarSigmoidDerivative;
___ACTIVATION[ActFunc.LOGSIGMOID]	= LogSigmoid;		___DERIVATIVE[ActFunc.LOGSIGMOID]	= LogSigmoidDerivative;
___ACTIVATION[ActFunc.LOGLOG]		= LogLog;			___DERIVATIVE[ActFunc.LOGLOG]		= LogLogDerivative;
___ACTIVATION[ActFunc.RELU]			= Relu;				___DERIVATIVE[ActFunc.RELU]			= ReluDerivative;
___ACTIVATION[ActFunc.RELUCAP]		= ReluCap;			___DERIVATIVE[ActFunc.RELUCAP]		= ReluCapDerivative;
___ACTIVATION[ActFunc.LEAKYRELU]	= LeakyRelu;		___DERIVATIVE[ActFunc.LEAKYRELU]	= LeakyReluDerivative;
___ACTIVATION[ActFunc.PRELU]		= PRelu;			___DERIVATIVE[ActFunc.PRELU]		= PReluDerivative;
___ACTIVATION[ActFunc.ELU]			= Elu;				___DERIVATIVE[ActFunc.ELU]			= EluDerivative;
___ACTIVATION[ActFunc.GELU]			= Gelu;				___DERIVATIVE[ActFunc.GELU]			= GeluDerivative;
___ACTIVATION[ActFunc.SELU]			= Selu;				___DERIVATIVE[ActFunc.SELU]			= SeluDerivative;
___ACTIVATION[ActFunc.CELU]			= Celu;				___DERIVATIVE[ActFunc.CELU]			= CeluDerivative;
___ACTIVATION[ActFunc.SWISH]		= Swish;			___DERIVATIVE[ActFunc.SWISH]		= SwishDerivative;
___ACTIVATION[ActFunc.ELISH]		= Elish;			___DERIVATIVE[ActFunc.ELISH]		= ElishDerivative;
___ACTIVATION[ActFunc.SOFTSIGN]		= SoftSign;			___DERIVATIVE[ActFunc.SOFTSIGN]		= SoftSignDerivative;		
___ACTIVATION[ActFunc.SOFTPLUS]		= SoftPlus;			___DERIVATIVE[ActFunc.SOFTPLUS]		= SoftPlusDerivative;	
___ACTIVATION[ActFunc.SOFTCLIP]		= SoftClipping;		___DERIVATIVE[ActFunc.SOFTCLIP]		= SoftClippingDerivative;
___ACTIVATION[ActFunc.SOFTEXP]		= SoftExponential;	___DERIVATIVE[ActFunc.SOFTEXP]		= SoftExponentialDerivative;
___ACTIVATION[ActFunc.SINC]			= Sinc;				___DERIVATIVE[ActFunc.SINC]			= SincDerivative;
___ACTIVATION[ActFunc.GAUSSIAN]		= Gaussian;			___DERIVATIVE[ActFunc.GAUSSIAN]		= GaussianDerivative;		
___ACTIVATION[ActFunc.SQRBF]		= SQRBF;			___DERIVATIVE[ActFunc.SQRBF]		= SQRBFDerivative;
___ACTIVATION[ActFunc.ISRU]			= ISRU;				___DERIVATIVE[ActFunc.ISRU]			= ISRUDerivative;
___ACTIVATION[ActFunc.ISRLU]		= ISRLU;			___DERIVATIVE[ActFunc.ISRLU]		= ISRLUDerivative;
___ACTIVATION[ActFunc.SQNL]			= SQNL;				___DERIVATIVE[ActFunc.SQNL]			= SQNLDerivative;			
___ACTIVATION[ActFunc.BENT]			= BentIdentity;		___DERIVATIVE[ActFunc.BENT]			= BentIdentityDerivative;	
___ACTIVATION[ActFunc.BINARY]		= BinaryStep;		___DERIVATIVE[ActFunc.BINARY]		= BinaryStepDerivative;	
___ACTIVATION[ActFunc.ABSOLUTE]		= Absolute;			___DERIVATIVE[ActFunc.ABSOLUTE]		= AbsoluteDerivative;	
___ACTIVATION[ActFunc.THRESHOLD]	= Threshold;		___DERIVATIVE[ActFunc.THRESHOLD]	= ThresholdDerivative;
___ACTIVATION[ActFunc.HARDTANH]		= HardTanh;			___DERIVATIVE[ActFunc.HARDTANH]		= HardTanhDerivative;
___ACTIVATION[ActFunc.HARDSIGMOID]	= HardSigmoid;		___DERIVATIVE[ActFunc.HARDSIGMOID]	= HardSigmoidDerivative;
___ACTIVATION[ActFunc.HARDSHRINK]	= HardShrink;		___DERIVATIVE[ActFunc.HARDSHRINK]	= HardShrinkDerivative;
___ACTIVATION[ActFunc.HARDSWISH]	= HardSwish;		___DERIVATIVE[ActFunc.HARDSWISH]	= HardSwishDerivative;
___ACTIVATION[ActFunc.HARDELISH]	= HardElish;		___DERIVATIVE[ActFunc.HARDELISH]	= HardElishDerivative;

// Making these as Method. Executes faster. 
//  -> Otherwise would always create temporal method when calling.
for(var i = 0; i < ActFunc.length; i++) {
	___ACTIVATION[i] = method(undefined, ___ACTIVATION[i]);
	___DERIVATIVE[i] = method(undefined, ___DERIVATIVE[i]);
}	// Though this makes you cannot get name simply by "script_get_name".


/*____________________________________________________________________________________________________________________________________
*/

/// @func	activation_function_enum(function);
/// @desc	Returns enum-index from given function.
/// @desc	Mostly used when initializing mlp.
function activation_function_enum(funktion) {
	switch(funktion) {	
		case Identity:			case IdentityDerivative:			return ActFunc.IDENTITY;	
		case Tanh:				case TanhDerivative:				return ActFunc.TANH;	
		case TanhLecun:			case TanhLecunDerivative:			return ActFunc.TANHLECUN;		
		case TanhShrink:		case TanhShrinkDerivative:			return ActFunc.TANHSHRINK;	
		case ArcTan:			case ArcTanDerivative:				return ActFunc.ARCTAN;		
		case sin:			/*	case cos:*/							return ActFunc.SIN;
		case cos:				case CosDerivative:					return ActFunc.COS;
		case Sigmoid:			case SigmoidDerivative:				return ActFunc.SIGMOID;	
		case BipolarSigmoid:	case BipolarSigmoidDerivative:		return ActFunc.BISIGMOID;	
		case LogSigmoid:		case LogSigmoidDerivative:			return ActFunc.LOGSIGMOID;
		case LogLog:			case LogLogDerivative:				return ActFunc.LOGLOG;			
		case Relu:				case ReluDerivative:				return ActFunc.RELU;		
		case ReluCap:			case ReluCapDerivative:				return ActFunc.RELUCAP;	
		case LeakyRelu:			case LeakyReluDerivative:			return ActFunc.LEAKYRELU;	
		case PRelu:				case PReluDerivative:				return ActFunc.PRELU;		
		case Elu:				case EluDerivative:					return ActFunc.ELU;		
		case Gelu:				case GeluDerivative:				return ActFunc.GELU;
		case Selu:				case SeluDerivative:				return ActFunc.SELU;
		case Celu:				case CeluDerivative:				return ActFunc.CELU;
		case Elish:				case ElishDerivative:				return ActFunc.ELISH;				
		case Swish:				case SwishDerivative:				return ActFunc.SWISH;		
		case SoftSign:			case SoftSignDerivative:			return ActFunc.SOFTSIGN;	
		case SoftPlus:			case SoftPlusDerivative:			return ActFunc.SOFTPLUS;	
		case SoftClipping:		case SoftClippingDerivative:		return ActFunc.SOFTCLIP;	
		case SoftExponential:	case SoftExponentialDerivative:		return ActFunc.SOFTEXP;	
		case Sinc:				case SincDerivative:				return ActFunc.SINC;		
		case Gaussian:			case GaussianDerivative:			return ActFunc.GAUSSIAN;	
		case SQRBF:				case SQRBFDerivative:				return ActFunc.SQRBF;		
		case ISRU:				case ISRUDerivative:				return ActFunc.ISRU;		
		case ISRLU:				case ISRLUDerivative:				return ActFunc.ISRLU;		
		case SQNL:				case SQNLDerivative:				return ActFunc.SQNL;		
		case BentIdentity:		case BentIdentityDerivative:		return ActFunc.BENT;		
		case BinaryStep:		case BinaryStepDerivative:			return ActFunc.BINARY;		
		case Threshold:			case ThresholdDerivative:			return ActFunc.THRESHOLD;
		case Absolute:			case AbsoluteDerivative:			return ActFunc.ABSOLUTE;
		case HardTanh:			case HardTanhDerivative:			return ActFunc.HARDTANH;	
		case HardSigmoid:		case HardSigmoidDerivative:			return ActFunc.HARDSIGMOID;
		case HardShrink:		case HardShrinkDerivative:			return ActFunc.HARDSHRINK;
		case HardElish:			case HardElishDerivative:			return ActFunc.HARDELISH;				
		case HardSwish:			case HardSwishDerivative:			return ActFunc.HARDSWISH;	
		default: 
			PRINT("No known enum for given function. Returned index for 'Identity'."); 
			return ActFunc.IDENTITY;
	}
}

/*___________________________________________________________________________________________________________________________________
*/

/// @func	activation_function_list_all();
/// @desc	Returns string of all activation functions.
function activation_function_list() {
	var text = "| ";
	var iEnd = array_length(___ACTIVATION);
	for(var i = 0; i < iEnd; i++) {
		text +=  string(i) + " : " + activation_function_name(___ACTIVATION[i]) + " | ";
	}
	return text;
}

/*___________________________________________________________________________________________________________________________________
*/

/// @func	activation_function_list_array(index_array);
/// @desc	Returns string of activation functions.
function activation_function_list_array(index_array) {
	var text = "| ";
	var iEnd = array_length(index_array);
	var index;
	for(var i = 0; i < iEnd; i++) {
		index = index_array[i];
		text +=  string(i) + " : " + script_get_name(___ACTIVATION[index]) + " | ";
	}
	return text;
}

/*___________________________________________________________________________________________________________________________________
*/

/// @func	activation_function_name(enumIndex);
/// @desc	Returns activation function as string
function activation_function_name(enumIndex) {
	switch(enumIndex) {
		case ActFunc.IDENTITY:			return script_get_name(Identity);		
		case ActFunc.TANH:				return script_get_name(Tanh);			
		case ActFunc.TANHLECUN:			return script_get_name(TanhLecun);	
		case ActFunc.TANHSHRINK:		return script_get_name(TanhShrink);	
		case ActFunc.ARCTAN:			return script_get_name(ArcTan);		
		case ActFunc.SIN:				return script_get_name(sin);			
		case ActFunc.COS:				return script_get_name(cos);			
		case ActFunc.SIGMOID:			return script_get_name(Sigmoid);		
		case ActFunc.BISIGMOID:			return script_get_name(BipolarSigmoid);
		case ActFunc.LOGSIGMOID:		return script_get_name(LogSigmoid);	
		case ActFunc.LOGLOG:			return script_get_name(LogLog);		
		case ActFunc.RELU:				return script_get_name(Relu);			
		case ActFunc.RELUCAP:			return script_get_name(ReluCap);		
		case ActFunc.LEAKYRELU:			return script_get_name(LeakyRelu);		
		case ActFunc.PRELU:				return script_get_name(PRelu);			
		case ActFunc.ELU:				return script_get_name(Elu);			
		case ActFunc.GELU:				return script_get_name(Gelu);			
		case ActFunc.SELU:				return script_get_name(Selu);			
		case ActFunc.CELU:				return script_get_name(Celu);			
		case ActFunc.ELISH:				return script_get_name(Elish);			
		case ActFunc.SWISH:				return script_get_name(Swish);			
		case ActFunc.SOFTSIGN:			return script_get_name(SoftSign);		
		case ActFunc.SOFTPLUS:			return script_get_name(SoftPlus);		
		case ActFunc.SOFTCLIP:			return script_get_name(SoftClipping);	
		case ActFunc.SOFTEXP:			return script_get_name(SoftExponential);
		case ActFunc.SINC:				return script_get_name(Sinc);			
		case ActFunc.GAUSSIAN:			return script_get_name(Gaussian);		
		case ActFunc.SQRBF:				return script_get_name(SQRBF);			
		case ActFunc.ISRU:				return script_get_name(ISRU);			
		case ActFunc.ISRLU:				return script_get_name(ISRLU);			
		case ActFunc.SQNL:				return script_get_name(SQNL);			
		case ActFunc.BENT:				return script_get_name(BentIdentity);	
		case ActFunc.BINARY:			return script_get_name(BinaryStep);	
		case ActFunc.THRESHOLD:			return script_get_name(Threshold);		
		case ActFunc.ABSOLUTE:			return script_get_name(Absolute);		
		case ActFunc.HARDTANH:			return script_get_name(HardTanh);		
		case ActFunc.HARDSIGMOID:		return script_get_name(HardSigmoid);	
		case ActFunc.HARDSHRINK:		return script_get_name(HardShrink);	
		case ActFunc.HARDELISH:			return script_get_name(HardElish);		
		case ActFunc.HARDSWISH:			return script_get_name(HardSwish);	
		default: return "Unknown activation function.";
	}									
}

/*___________________________________________________________________________________________________________________________________
*/











































