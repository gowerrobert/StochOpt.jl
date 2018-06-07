function testerror(testprob::Prob, x)
return mean((prob.X'*x).*prob.y .>0);
end
