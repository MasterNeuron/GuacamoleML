

/// @func	___mlp_plus_genetic_algorithm(population, elitism, mutationism, mutationRate);
/// @desc	Uses genetic algorithm to update given population. Given population should already be arranged with Fitness-function. 
/// @param	{array}		population		Array of mlp's. (mlp_plus)	
/// @param	{real}		elitism			Which portion of elite population continues to next generation. Rest are childs of elite
/// @param	{real}		mutationism		Which portion of new generation will be mutated
/// @param	{real}		mutationAmount	How much mutate given individual
/// @param	{real}		mutationRate	Maximium amount for random mutation
function ___mlp_plus_genetic_algorithm(population, elitism, mutationism, mutationAmount, mutationRate) {
	var count = array_length(population);
	var sizeOf = buffer_sizeof(NumberType.DOUBLE);
	var buffer = buffer_create(count * sizeOf, buffer_fixed, sizeOf);
	
	// Get population mlp-indexes to buffer.
	for(var i = 0; i < count; i++) {
		buffer_write(buffer, NumberType.DOUBLE, population[i].mlp_index);
	}	buffer_seek(buffer, buffer_seek_start, 0);
	
	var pointer = buffer_get_address(buffer);
	
	// Update MLP's.
	var debug_success;
	debug_success = ___ext_genetic_population(pointer, count);
	debug_success = ___ext_genetic_use(elitism, mutationism, mutationAmount, mutationRate);
	debug_success = ___ext_genetic_clear();

	buffer_delete(buffer);
}