/// DEFAULT RANDOMIZING
/// These are for default randomizing functions for neural networks.
/// Starting values need to be randomized. Though sometimes different ways work better.
/// Here it's easier to change for all, if you want to use normal distribution curve etc.

/// @func	Random();
/// @desc	Default random value. 
function Random() {
	return random_range(0,1);
}

/// @func	RandomWeight();
/// @desc	Default random weights.
function RandomWeight() {
	return random_range(-.5,+.5);	// Many things affect which random Initializing value value works best. 
}

/// @func	RandomBias();
/// @desc	Default random biases.
function RandomBias() {
	return random_range(-.2,+.2);
}

/// @func	RandomFilter();
/// @desc	Default random filter values.
function RandomFilter() {
	return random_range(-1,+1);
}

/// @func	weighted_random(probabilities);
/// @desc	Returns index of given argument by weighted probability of given array. Array items with negative value are skipped.
/// @desc	This function is useful if you want to use neural network output array as probablities to choose from.
/// @param	{array}	probabilities
/// @return {int}	array position
function weighted_random(probabilities) {	
	// Get sum of all
	var sum = 0;
	var size = array_length(probabilities);
	for(var i = 0; i < size; i++) {
		sum += max(0, probabilities[i]);
	}
	
	// If all are <= 0, then no given probability distribution -> equally likely.
	if (sum <= 0) {
		return irandom(size-1);	// irandom is inclusive, so have to remove 1 entry. 
	}
	
	// Otherwise use distribution
	var rnd = random(sum);
	for(var i = 0; i < size; i++) {
		if (probabilities[i] <= 0)
			continue;
		if (rnd < probabilities[i]) 
			return i;
		rnd -= probabilities[i];
	}
	return i;
}
