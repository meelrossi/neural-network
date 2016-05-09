function resp = or (inputs)
    inputs_count = rows(inputs);
    for i = 1 : inputs_count
        resp(i) = sum(inputs(i, :)) > 0;
    end
    resp = resp';
end