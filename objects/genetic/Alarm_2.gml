for(i = 0; i < array_length(elements); i++)
		{
		// RUN THE RUN FUNCTION FOR EVERY ELEMENT
		elements[i].Run();
		}
//alarm[0] = 1;



////////////////////

ff = function(A, B) {
	// return (B.assets * 100 + B.maxassets * 90) - (A.assets * 100 + A.maxassets * 90);
	return (B.rating - A.rating);
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

with (panel)
{
	if (self.elid = other.bestel.elid)
	{
	image_alpha = 1;
	//show_debug_message(string(image_alpha) + " " + string(irandom_range(1,100)));
	}
	else
	{
	image_alpha = 0.07;
	}
}

/*
// SET A TEMPORARY VARIABLE
var n = 1;
// REMOVE ALL PAST BUY AND SELL SIGNALS FROM THE ARRAY
repeat(chart.size)
{
// SET THE BUYSELL TO 0
global.buysell[n] = 0;
// INCREASE THE COUNTER
n = n + 1;
}
*/

// MUTATIONISM
mimin = 1;
mimax = 20;

// MUTATION AMOUNT
mamin = 1;
mamax = 20;

// MUTATION RATE
rmin = 1; //30
rmax = 20; //30

// ELITISM
emin = 2;
emax = 10;

// DROP THE BEAT! MUTATE
___mlp_grid_genetic_algorithm(elbrains, irandom_range(emin,emax)/100, irandom_range(mimin,mimax)/100, irandom_range(mamin,mamax)/100, irandom_range(rmin,rmax)/100); //.03, .6, .3, .2

// RESTART THE ELEMENTS
for(i = 0; i < array_length(elements); i++) {
	elements[i].Restart();
}

alarm[2] = 1;