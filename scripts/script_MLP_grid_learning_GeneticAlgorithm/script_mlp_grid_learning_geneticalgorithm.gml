
/// @func	___mlp_grid_genetic_algorithm(population, elitism, mutationism, mutationRate);
/// @desc	Uses genetic algorithm to update given population. Given population should already be arranged with Fitness-function. 
/// @desc	This function is not a "ground truth", but just one simple way of implementing genetic algorithm. 
/// @param	{array}		population		Array of mlp's. (mlp_grid)	
/// @param	{real}		elitism			Which portion of elite population continues to next generation. Rest are childs of elite
/// @param	{real}		mutationism		Which portion of new generation will be mutated
/// @param	{real}		mutationAmount	How much mutate given individual
/// @param	{real}		mutationRate	Maximium amount for random mutation
function ___mlp_grid_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) {
	// Genetic algorithm does three things:
	// 1) Selection:	choose best for elite
	// 2) Crossover:	make children from best
	// 3) Mutation:		tiny changes to children
	
	var i, j, k, iEnd, jEnd, kEnd; 
	var a, b, c;
	var populationCount, eliteCount;
	var child, parent, parentA, parentB;

	// 1) Selection
	// Take portion of population as Elite for next generation.
	// Assumes array has arranged Best-Worst already. Fitness-function is the how you arrange them, the way varies case-to-case.
	populationCount = array_length(population)
	eliteCount = max(1, ceil(populationCount * elitism));

	// 2) Cross-over
	// Make childs of elite population. Children copy randomly parts from parents.
	// In this example children has two parents, but you could have more.
	for(c = eliteCount; c < populationCount; c++) {
		a = irandom(eliteCount-1);
		b = irandom(eliteCount-1);
		while((a == b) && (eliteCount > 1)) {
			b = irandom(eliteCount-1);
		}
		parentA = population[a];
		parentB = population[b];
		child = population[c];
	
		// Select all parameters randomly from the parents.
		iEnd = child.layerCount;
		for(i = 1; i < iEnd; i++) {
			jEnd = child.layerSizes[i];
			kEnd = child.layerSizes[i-1];
			// Select activation function from either parent
			parent = choose(parentA, parentB);
			child.ActivationFunction[i] = parent.ActivationFunction[i];
			child.activation_parameter[i] = parent.activation_parameter[i];
			for(j = 0; j < jEnd; j++) {
				// Go through all weights & biases
				parent = choose(parentA, parentB);
				child.bias[@i][@j] = parent.bias[i][j];
				for(k = 0; k < kEnd; k++) {
					parent = choose(parentA, parentB);
					child.weights[@i][# j, k] = parent.weights[i][# j, k];
				}
			}
		}
	}

	// 3) Mutation
	// Mutate some of the childs with given mutation rate
	// To keep it simple we allow mutations happen to same specimen again.
	repeat((populationCount - 1) * mutationism) {
		c = irandom_range(eliteCount, populationCount-1);	// Save elite from mutations.
		child = population[c];
		iEnd = child.layerCount;
		for(i = 1; i < iEnd; i++) {
			jEnd = child.layerSizes[i];
			kEnd = child.layerSizes[i-1];
			
			// Mutate weights
			repeat(max(1, mutationAmount * jEnd * kEnd)) {
				j = irandom(jEnd-1);
				k = irandom(kEnd-1);
				child.weights[@i][# j, k] += random_range(-mutationRate, +mutationRate);
			}
			// Mutate biases
			repeat(max(1, mutationAmount * jEnd)) {
				j = irandom(jEnd-1);
				child.bias[@i][@j] += random_range(-mutationRate, +mutationRate);
			}
		}
	}
}