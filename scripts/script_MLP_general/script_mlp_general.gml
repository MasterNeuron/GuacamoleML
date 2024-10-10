
function mlp_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) {
	if (!is_array(population))
	or (!is_struct(population[0])) {
		return;
	}
	switch(population[0].type) {
		case nn.ARRAY:	___mlp_array_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate);	break;
		case nn.GRID:	___mlp_grid_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate);	break;
		case nn.PLUS:	___mlp_plus_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate);	break;
		default: throw("Unknown MLP type in mlp_genetic_algorithm.");
	}
}

/// @func	___mlp_acceptable_precision(precision);
/// @desc	Forces input to be one of acceptable accuracies. Default is 64bit.
/// @param	{enum}	precision
function ___mlp_acceptable_precision(precision) {
	return ((precision == NumberType.HALF) || (precision == NumberType.FLOAT)) ? precision : NumberType.DOUBLE;
}

/// @func	is_mlp(mlp);
/// @desc	Checks wether is mlp, not including simple ones.
function is_mlp(mlp) {	// Don't consider simple ones.
	var instOf = instanceof(mlp);
	return (instOf == script_get_name(mlp_array) // More easily find error if happen to change MLP names.
	|| instOf == script_get_name(mlp_grid) 
	|| instOf == script_get_name(mlp_plus));
}

/// @func	___mlp_activation_functions(funcs);
/// @desc	Help function to create full index-array from given varying inputs. 
/// @desc	Assumes this is called in initialization of MLP, not mean to be used anywhere else.
/// @desc	If given array is too short, last item is repeated. If undefined, default is Tanh.
function ___mlp_activation_functions(funcs) {
	var activationFunctions, func, index;
	// Use default
	if (is_undefined(funcs)) {
		activationFunctions = array_create(layerCount, ActFunc.TANH);
	// All layers have same func.
	} else if (!is_array(funcs)) {
		index = activation_function_enum(funcs);
		index = (index == -1) ? 0 : index;
		activationFunctions = array_create(layerCount, index);
	// Layers have differing funcs.
	} else {
		var i, maxIndex = array_length(funcs)-1;
		activationFunctions = array_create(layerCount);
		for(i = 0; i < layerCount; i++) {
			func = funcs[min(i, maxIndex)];			// Uses array items or repeats the last one.
			index = activation_function_enum(func);
			index = (index == -1) ? 0 : index;
			activationFunctions[i] = index;	
		}
	}
	return activationFunctions;	
}

