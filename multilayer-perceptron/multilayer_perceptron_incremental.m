
function smart_nets = multilayer_perceptron_incremental(nets, t, err, g, g_inv, n ,b)
    inputs = t{1}; % matrix[inputs_count][input_size]
    inputs_count = rows(inputs);
    input_size = columns(inputs);
    s = t{2}; % matrix[inputs_count][s_size]
    s_size = columns(s);

	nets_count = size(nets)(2);

	% forward step
	in_layer = inputs;
	for i = 1 : nets_count
		layer = [ones(inputs_count, 1).*(-1) in_layer];
		V{i} = g(layer * nets{i}, b);
		in_layer = V{i};
	end


	

end