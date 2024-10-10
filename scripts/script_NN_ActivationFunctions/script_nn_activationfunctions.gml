#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________________________________________

	List of activation functions and their derivatives.
	Take a look wikipedia page, where you can find most of these:
		https://en.wikipedia.org/wiki/Activation_function
		https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
		https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
		
	It is hard to say describe the functions, so look Activation-example, which shows them as graphs :)
____________________________________________________________________________________________________________________________________
*/
#endregion


/*________________________________________________________________________________________________________________________________
*/

/// @func	Identity(input);
/// @desc	Pass-through function. If method-variable is needed but want nothing to happen.
/// @desc	Same as identity.
function Identity(input) {
	return input;
}
/// @func	IdentityDerivative(input);
function IdentityDerivative(input) {
	return 1;
}
	
/*________________________________________________________________________________________________________________________________
*/

/// @func	Tanh(input);
/// @desc	Squishes value to be between -1 and +1 as S-curve. Similiar to Sigmoid
function Tanh(input) {
	return ((2 / (1 + exp(-2 * input))) - 1);
}
/// @func	TanhDerivative(input);
function TanhDerivative(input) {
	return (1 - sqr(Tanh(input)));
}

/*________________________________________________________________________________________________________________________________
*/
/// @func	TanhLecun(input);
/// @desc	See : http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
function TanhLecun(input) {
	return 1.7159 * Tanh(input * 2/3);
}
/// @func	TanhLecunDerivative(input);
function TanhLecunDerivative(input) {
	return 4.57573 / sqr(exp(-2*input/3) + exp(2*input/3));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	TanhShrink(input);
/// @desc	
function TanhShrink(input) {
	return input - Tanh(input);
}
/// @func	TanhShrinkDerivative(input);
function TanhShrinkDerivative(input) {
	return sqr(Tanh(input));
}

/*________________________________________________________________________________________________________________________________
*/
/// @func	ArcTan(input);
/// @desc	
function ArcTan(input) {
	return arctan(input);
}
/// @func	ArcTanDerivative(input);
function ArcTanDerivative(input) {
	return 1 / (sqr(input) + 1);
}

/*________________________________________________________________________________________________________________________________
*/

/* Use native cos function as activation function. Derivative is (-sin(x)) so it needs to have own function though.*/

/// @func	CosDerivative(input);
/// @desc	
function CosDerivative(input) {
	return -sin(input);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Sigmoid(input);
/// @desc	a.k.a Logistig or Soft step.
/// @desc	Squishes value to be between 0 and +1 as S-curve. Similiar to Tanh
function Sigmoid(input) {
	return (1 / (1 + exp(-input)));
}
/// @func	SigmoidDerivative(input);
function SigmoidDerivative(input) {
	input = Sigmoid(input);			// Get result of Sigmoid so we don't need to calculate it twice
	return (input * (1 - input));	// Derivative of sigmoid is: Sigmoid * (1 - Sigmoid)
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	BipolarSigmoid(input);
/// @desc	
function BipolarSigmoid(input) {
	input = exp(-input);
	return (1 - input) / (1 + input);
}
/// @func	BipolarSigmoidDerivative(input);
function BipolarSigmoidDerivative(input) {
	input = exp(input);
	return 2 * input / sqr(input + 1);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	LogSigmoid(input);
/// @desc	
function LogSigmoid(input) {
	return ln(Sigmoid(input));
}
/// @func	LogSigmoidDerivative(input);
function LogSigmoidDerivative(input) {
	return 1 / (exp(input) + 1);	
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	LogLog(input);
/// @desc	Complementary Log-Log.
function LogLog(input) {
	return 1 - exp(-exp(input));
}
/// @func	LogLogDerivative(input);
function LogLogDerivative(input) {
	return exp(input - exp(input));	
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Relu(input);
/// @desc	Rectified Linear Unit ReLU.
function Relu(input) {
	return max(0, input);
}
/// @func	ReluDerivative(input);
function ReluDerivative(input) {
	return (input > 0);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	ReluCap(input, parameter);
/// @desc	Rectified Linear Unit ReLU with cap.
function ReluCap(input, parameter) {
	return min(max(0, input), parameter);
}
/// @func	ReluCapDerivative(input, parameter);
function ReluCapDerivative(input, parameter) {
	return (input > 0) && (input < parameter);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	LeakyRelu(input);
/// @desc	Similiar to Relu, but doesn't hard cap negative input.
function LeakyRelu(input) {
	return max(input*.1, input);	
}
/// @func	LeakyReluDerivative(input);
function LeakyReluDerivative(input) {
	return (input > 0) ? 1 : .1;
}		

/*________________________________________________________________________________________________________________________________
*/

/// @func	PRelu(input, parameter);
/// @desc	Parameteric rectified linear unit
function PRelu(input, parameter) {
	return max(input * parameter, input);	
}
/// @func	PReluDerivative(input, parameter);
function PReluDerivative(input, parameter) {
	return (input > 0) ? 1 : parameter;
}		

/*________________________________________________________________________________________________________________________________
*/

/// @func	Elu(input, parameter);
/// @desc	Exponential linear unit
function Elu(input, parameter) {
	return (input > 0) ? input : parameter * (exp(input) - 1);
}
/// @func	EluDerivative(input, parameter);
function EluDerivative(input, parameter) {
	return ((input > 0) || ((input == 0) && (parameter == 1))) ? 1 : parameter * exp(input);
}	

/*________________________________________________________________________________________________________________________________
*/

/// @func	Gelu(input);
/// @desc	Gaussian Error Linear Unit
/// @desc	Just approximation, because real one uses Error-function (Erf).
/// @desc	Approximation taken from here: https://github.com/hendrycks/GELUs
function Gelu(input) {
	return Sigmoid(1.702 * input) * input;
}
/// @func	GeluDerivative(input);
/// @desc	Derivative calculated from approximation. 
/// @desc	Calculated here: https://www.wolframalpha.com/input/?i=derivative+of+sigmoid%281.702+*+x%29+*+x
function GeluDerivative(input) {
	var expInput = exp(1.702*input);	// Precalculate.
	return (expInput * (1.702*input + expInput + 1)) / sqr(expInput + 1); 
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Selu(input);
/// @desc	Scaled exponential linear unit.
function Selu(input) {
	//return 1.0507 * ((input >= 0) ? input : 1.67326 * (exp(input) - 1));
	return 1.0507 * (max(0, input) + min(0, 1.67326 * (exp(input) - 1)));
}
/// @func	SeluDerivative(input);
function SeluDerivative(input) {
	return 1.0507 * ((input >= 0) ? 1 : 1.67326 * exp(input));
}	

/*________________________________________________________________________________________________________________________________
*/

/// @func	Celu(input, parameter);
/// @desc	Continuosly differentiable exponential linear unit.
function Celu(input, parameter) {
	return max(0, input) + min(0, parameter * (exp(input / parameter) - 1));
}
/// @func	CeluDerivative(input, parameter);
function CeluDerivative(input, parameter) {
	var expInputPar = exp(input / parameter);
	if ((input > 0) || (parameter * expInputPar >= parameter)) {
		return 1;
	} else if ((input <= 0) || (parameter * expInputPar < parameter)) {
		return expInputPar;
	} else {
		return expInputPar + 1;
	}
}	

/*________________________________________________________________________________________________________________________________
*/

/// @func	Elish(input);
/// @desc	Exponential linear Squashing, combination of Elu and Sigmoid.
/// @desc	https://paperswithcode.com/method/elish
function Elish(input) {
	return (input >= 0) 
		? (input / (1 + exp(-input))) 
		: ((exp(input) - 1) / (1 + exp(-input)));
}
/// @func	ElishDerivative(input);
function ElishDerivative(input) {
	var expInput = exp(input);
	return (input >= 0) 
		? ((expInput * (input + expInput + 1)) / sqr(expInput + 1)) 
		: expInput * (2 * expInput + exp(input * 2) - 1) / sqr(expInput + 1);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Swish(input);
/// @desc	Sigmoid linear unit, a.k.a Silu, Sigmoid shrinkage, Swish-1
function Swish(input) {
	return input / (1 + exp(-input));	// input * Sigmoid(input);
}
/// @func	SwishDerivative(input);
function SwishDerivative(input) {
	return (1 + exp(-input) + input * exp(-input)) / sqr(1 + exp(-input));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	SoftSign(input);
/// @desc	
function SoftSign(input) {
	return input / (1 + abs(input));
}
/// @func	SoftSignDerivative(input);
/// @desc	
function SoftSignDerivative(input) {
	return 1 / sqr(1 + abs(input));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	SoftPlus(input);
/// @desc	Aka ElliotSig.
function SoftPlus(input) {
	return ln(1 + exp(input));
}
/// @func	SoftPlusDerivative(input);
/// @desc	
function SoftPlusDerivative(input) {
	return 1 / (1 + exp(-input));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	SoftClipping(input, parameter);
/// @desc	
function SoftClipping(input, parameter) {
	return ln((1 + exp(parameter * input)) / (1 + exp(parameter * (input - 1)))) / parameter;
}
/// @func	SoftClippingDerivative(input, parameter);
/// @desc	Used WolframeAlpha for calculating derivative.
function SoftClippingDerivative(input, parameter) {
	var expPar = exp(parameter);
	var expParInput = exp(parameter * input);
	return ((expPar - 1) * expParInput) / ((expParInput + 1) * (expParInput + expPar));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	SoftExponential(input, parameter);
/// @desc	
function SoftExponential(input, parameter) {
	if (parameter < 0) {
		return -(ln(1 - parameter * (input + parameter))) / parameter;
	} else if (parameter == 0) {
		return input;
	} else {
		return (exp(parameter * input) - 1) / parameter + parameter;
	}
}
/// @func	SoftExponentialDerivative(input, parameter);
/// @desc	Used WolframeAlpha for calculating derivative.
function SoftExponentialDerivative(input, parameter) {
	if (parameter < 0) {
		return 1 / (1 - parameter * (parameter * input));
	} else {
		return exp(parameter * input);
	}
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Sinc(input);
/// @desc	
function Sinc(input) {
	return (input == 0) ? 1 : (sin(input)/input);
}
/// @func	SincDerivative(input);
/// @desc	
function SincDerivative(input) {
	return (input == 0) ? 0 : (cos(input)/input - sin(input)/sqr(input));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Gaussian(input);
/// @desc	
function Gaussian(input) {
	return exp(-sqr(input));
}
/// @func	GaussianDerivative(input);
/// @desc	
function GaussianDerivative(input) {
	return -2 * input * exp(-sqr(input));
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	SQRBF(input);
/// @desc	SQ-RBF.
function SQRBF(input) {
	if (abs(input) <= 1) {
		return 1 - sqr(input) / 2;
	} else if (abs(input) >= 2) {
		return 0;
	} else {
		return sqr(2 - abs(input)) / 2;
	}
}
/// @func	SQRBFDerivative(input);
/// @desc	
function SQRBFDerivative(input) {
	if (abs(input) <= 1) {
		return -input;
	} else if (abs(input) >= 2) {
		return 0;
	} else {
		return input - 2 * sign(input);
	}
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	ISRU(input, parameter);
/// @desc	Inverse Square Root Unit
function ISRU(input, parameter) {
	return input / sqrt(1 + parameter * sqr(input));
}
/// @func	ISRUDerivative(input, parameter);
/// @desc
function ISRUDerivative(input, parameter) {
	return power(1 / sqrt(1 + parameter * sqr(input)), 3);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	ISRLU(input, parameter);
/// @desc	Inverse Square Root Linear Unit
function ISRLU(input, parameter) {
	return (input >= 0) ? input : (input / sqrt(1 + parameter * sqr(input)));
}
/// @func	ISRLUDerivative(input, parameter);
/// @desc
function ISRLUDerivative(input, parameter) {
	return (input >= 0) ? 1 : power(1 / sqrt(1 + parameter * sqr(input)), 3);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	SQNL(input);
/// @desc	Square nonlinearity
function SQNL(input) {
	if (input > 2) {
		return 1;
	} else if (input < -2) {
		return -1;
	} else if (input < 0) {
		return input + sqr(input)/4;
	} else {
		return input - sqr(input)/4;
	}
}
/// @func	SQNLDerivative(input);
/// @desc	
function SQNLDerivative(input) {
	return (abs(input) > 2) ? 0 : ((input < 0) ? (1+input*.5) : (1-input*.5));
}

/*________________________________________________________________________________________________________________________________
*/
/// @func	BentIdentity(input);
/// @desc	
function BentIdentity(input) {
	return (sqrt(sqr(input) + 1) - 1) / 2 + input;
	
}
/// @func	BentIdentityDerivative(input);
/// @desc	
function BentIdentityDerivative(input) {
	return input / (2 * sqrt(sqr(input) + 1)) + 1;
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	BinaryStep(input);
/// @desc
function BinaryStep(input) {
	return (input >= 0);
}
/// @func	BinaryStepDerivative(input);
/// @desc	Binary step does not work for gradient descent. Doesn't have well-behaving derivative.
function BinaryStepDerivative(input) {
	return 0; // Actually should be: "(input != 0) ? 0 : undefined;" But let have 0 to not cause errors.
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Threshold(input, parameter);
/// @desc	Same as binary-step, but definable threshold for activation.
function Threshold(input, parameter) {
	return (input >= parameter);
}
/// @func	ThresholdDerivative(input, parameter);
/// @desc	Threshold  does not work for gradient descent. Doesn't have well-behaving derivative.
function ThresholdDerivative(input, parameter) {
	return 0; // Actually should be: "(input != parameter) ? 0 : undefined;" But let have 0 to not cause errors.
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	Absolute(input);
function Absolute(input) {
	return abs(input);
}
/// @func	AbsoluteDerivative(input);
function AbsoluteDerivative(input) {
	return sign(input);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	HardTanh(input);
/// @desc
function HardTanh(input) {
	return clamp(input, -1, 1);
}
/// @func	HardTanhDerivative(input);
/// @desc
function HardTanhDerivative(input) {
	return (input <= 1) && (input >= -1);
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	HardSigmoid(input);
/// @desc
function HardSigmoid(input) {
	return (input <= -3) ? 0 : ((input >= 3) ? 1 : (input/6 + .5));
}
/// @func	HardSigmoidDerivative(input);
/// @desc
function HardSigmoidDerivative(input) {
	return ((input > -3) && (input < 3)) / 6;
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	HardShrink(input, parameter);
/// @desc
function HardShrink(input, parameter) {
	return input * ((input > parameter) || (input < -parameter));
}
/// @func	HardShrinkDerivative(input, parameter);
/// @desc
function HardShrinkDerivative(input, parameter) {
	return (input > parameter) || (input < -parameter);
}
/*________________________________________________________________________________________________________________________________
*/

/// @func	HardElish(input);
/// @desc	Variant of Elish. 
/// @desc	https://paperswithcode.com/method/elish
function HardElish(input) {
	return (input >= 0) 
		? input * max(0, min(1, (input + 1) * .5))
		: ((exp(input) - 1) * max(0, min(1, (input + 1) * .5)));
}
/// @func	HardElishDerivative(input);
function HardElishDerivative(input) {
	if (input >= 0) {
		return (input > 1) ? 1 : (input + .5);
	} else {
		return (input < -1) ? 0 : exp(input) * (.5 * input + 1) - .5; 
	}
}

/*________________________________________________________________________________________________________________________________
*/

/// @func	HardSwish(input);
/// @desc
function HardSwish(input) {
	return (input <= -3) ? 0 : ((input >= 3) ? input : input * (input + 3) / 6);
}
/// @func	HardSwishDerivative(input);
/// @desc
function HardSwishDerivative(input) {
	return (input <= -3) ? 0 : ((input >= 3) ? 1 : (2 * input + 3) / 6);
}

/*________________________________________________________________________________________________________________________________
*/