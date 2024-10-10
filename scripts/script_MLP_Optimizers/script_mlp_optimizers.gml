#region /// INFORMATION ABOUT SCRIPT
/*
____________________________________________________________________________________________________

	LIST OF AVAILABLE GRADIENT DESCENT OPTIMIZERS
	
	To allow initializing optimizer for all MLP types with same name. 
	Example : mlp.Optimizer(Nesterov, .8);
	
	Also you can define during mlp creation like:
	 - mlp = new mlp_array(layers).Optimizer(Stochastic);

____________________________________________________________________________________________________
*/
#endregion


///////////////////////////////////////////////////////////////////////////////////////
///																					///
///		These assume being called by mlp! -> "mlp.Optimizer(Stochastic);"			///
///																					///
///////////////////////////////////////////////////////////////////////////////////////
function Stochastic() {
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_stochastic(self);
		case nn.GRID:	return new ___optimizer_grid_stochastic(self);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.STOCHASTIC, 0);
		default: throw("Initializing Stochastic-optimizer: Unknown MLP type.");
	}
}

function Momentum(momentumRate) {
	momentumRate = is_undefined(momentumRate) ? .5 : momentumRate;
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_momentum(self, momentumRate);
		case nn.GRID:	return new ___optimizer_grid_momentum(self, momentumRate);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.MOMENTUM, momentumRate);
		default: throw("Initializing Momentum-optimizer: Unknown MLP type.");
	}
}

function Nesterov(momentumRate) {
	momentumRate = is_undefined(momentumRate) ? .5 : momentumRate;
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_nesterov(self, momentumRate);
		case nn.GRID:	return new ___optimizer_grid_nesterov(self, momentumRate);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.NESTEROV, momentumRate);
		default: throw("Initializing Nesterov-optimizer: Unknown MLP type.");
	}
}

function Adam() {
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_adam(self);
		case nn.GRID:	return new ___optimizer_grid_adam(self);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.ADAM, 0);
		default: throw("Initializing Adam-optimizer: Unknown MLP type.");
	}
}

function AdaGrad() {
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_adagrad(self);
		case nn.GRID:	return new ___optimizer_grid_adagrad(self);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.ADAGRAD, 0);
		default: throw("Initializing Adagrad-optimizer: Unknown MLP type.");
	}
}

function AdaDelta(decayRate) {
	decayRate = is_undefined(decayRate) ? .95 : decayRate;
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_adadelta(self, decayRate);
		case nn.GRID:	return new ___optimizer_grid_adadelta(self, decayRate);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.ADADELTA, decayRate);
		default: throw("Initializing Adadelta-optimizer: Unknown MLP type.");
	}
}

function AdaMax() {
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_adamax(self);
		case nn.GRID:	return new ___optimizer_grid_adamax(self);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.ADAMAX, 0);
		default: throw("Initializing Adamax-optimizer: Unknown MLP type.");
	}
}

function Nadam() {
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_nadam(self);
		case nn.GRID:	return new ___optimizer_grid_nadam(self);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.NADAM, 0);
		default: throw("Initializing Nadam-optimizer: Unknown MLP type.");
	}
}

function RMSprop(decayRate) {
	decayRate = is_undefined(decayRate) ? .9 : decayRate;
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_rmsprop(self, decayRate);
		case nn.GRID:	return new ___optimizer_grid_rmsprop(self, decayRate);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.RMSPROP, decayRate);
		default: throw("Initializing RMSprop-optimizer: Unknown MLP type.");
	}
}

function AMSgrad() {
	switch(type) {
		case nn.ARRAY:	return new ___optimizer_array_amsgrad(self);
		case nn.GRID:	return new ___optimizer_grid_amsgrad(self);
		case nn.PLUS:	return new ___optimizer_plus(self, OptimizerType.AMSGRAD, 0);
		default: throw("Initializing AMSgrad-optimizer: Unknown MLP type.");
	}
}