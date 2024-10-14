// CREATE A FITNESS FUNCTION
ff = function(A, B) {
	// B-A HIGHER IS BETTER
	// A-B LOWER IS BETTER
	return ((A.rating) - (B.rating));
}

// SORT THEM BASED ON THAT FUNCTION
array_sort(elements, ff);

// TELL THE CURRENT TOP ELEMENT THAT IT IS NOT THE BEST ANYMORE
bestel.top = 0;
// SELECT THE NEW BEST ELEMENT
bestel = elements[0];
// TELL THAT ELEMENT THAT IT IS THE BEST ONE
bestel.top = 1;

// SAVE THE ELEMENT BRAINS
for(i = 0; i < array_length(elements); i++) {
	elbrains[i] = elements[i].mlp;
}

// MUTATIONISM
mimin = 30;
mimax = 30;

// MUTATION AMOUNT
mamin = 30;
mamax = 30;

// MUTATION RATE
rmin = 30;
rmax = 30;

// ELITISM
emin = 3;
emax = 3;

// DROP THE BEAT! MUTATE
___mlp_grid_genetic_algorithm(elbrains, irandom_range(emin,emax)/100, irandom_range(mimin,mimax)/100, irandom_range(mamin,mamax)/100, irandom_range(rmin,rmax)/100); //.03, .6, .3, .2

// RESTART THE ELEMENTS
for(i = 0; i < array_length(elements); i++) {
	elements[i].Restart();
}

// BRING BACK THE PROCESS TO THE ALARM 0 LOOP
alarm[0] = 1;