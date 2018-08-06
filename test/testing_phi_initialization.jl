w = rand(prob.numfeatures)#rand(prob.numfeatures)
Xx = prob.X'*w;
yXx = prob.y.*Xx;
probs = logistic_phi(yXx);
probs[:] = probs.*(1-probs);
# recording the max phi
# println(probs)
println("randn mean phi val: ", mean(probs))
println("randn std phi val: ", std(probs))

w = zeros(prob.numfeatures);#rand(prob.numfeatures)
Xx = prob.X'*w;
yXx = prob.y.*Xx;
probs = logistic_phi(yXx);
probs[:] = probs.*(1-probs);
# recording the max phi
# println(probs)
println("zero mean phi val: ", mean(probs))
println("zero std phi val: ", std(probs))


w = ones(prob.numfeatures);#rand(prob.numfeatures)
Xx = prob.X'*w;
yXx = prob.y.*Xx;
probs = logistic_phi(yXx);
probs[:] = probs.*(1-probs);
# recording the max phi
# println(probs)
println("ones mean phi val: ", mean(probs))
println("ones std phi val: ", std(probs))

density = 0;
for i = 1:prob.numdata
    density += nnz(prob.X[:, i])/prob.numdata;
end
println("Feature density: ",density/prob.numfeatures)