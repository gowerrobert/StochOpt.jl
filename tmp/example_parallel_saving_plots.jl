## Toy example saving figures in parallel

using Distributed
addprocs(4)

@everywhere begin # this part will be available on all CPUs
    path = "/home/nidham/phd/StochOpt.jl/"; # Be carefull about the full path here
    using Plots

    pyplot() # No problem with pyplot when called in @everywhere statement
end

@everywhere function save_plot_parallel(i, path)
    x = 1:10;
    y = x .^ i;
    plt = plot(x, y, label=string(i));
    println("$(path)figures/$(string(i)).pdf")
    savefig(plt, "$(path)figures/$(string(i)).pdf");
end

## Toy example shows that savefig works in @distributed for loops
@sync @distributed for i=1:3
    println("Iteration: ", i);
    save_plot_parallel(i, path);
end