//if (global.simstep < global.simstepend)
if (global.totalsteps > 0)
{
	for(i = 0; i < array_length(elements); i++)
		{
		// RUN THE RUN FUNCTION FOR EVERY ELEMENT
		elements[i].Run();
		}
	// INCREASE THE SIM STEP
	global.simstep = irandom_range(1,global.simstepend - 1);
	global.totalsteps = global.totalsteps - 1;
	global.s = global.s + 1;
	// REPEAT
	alarm[0] = 1;
}
else
{
// WE REACHED THE END - START FROM THE BEGINING
global.simstep = 1;
global.s = 0;
global.totalsteps = global.maxsteps;
// INCREASE THE GENERATION BY ONE
global.generation = global.generation + 1;
// ACTIVATE THE EVOLUTION
alarm[1] = 1;
}