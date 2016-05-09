function answer = test_incremental (ft)
    % inputs = [1 1; 1 0; 0 1; 0 0];
    inputs = [1 1 1; 1 0 1; 0 1 1; 0 0 1; 0 1 0; 1 0 0; 1 1 0; 0 0 0];
    inputs_count = rows(inputs);
    outputs = ft(inputs);
    t{1} = inputs;
    t{2} = outputs;
    n = 0.5;
    net = simple_perceptron_incremental(0, t, @step, n);

    for i = 1 : inputs_count
        input = [-1 inputs(i, :)];
        output = step(input * net);
        fprintf('input \n');
        input
        fprintf('output \n');
        output
    end
end