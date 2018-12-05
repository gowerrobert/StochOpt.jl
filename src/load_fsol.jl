function load_fsol!(options, prob)
    if(options.exacterror)
        try # getting saved optimal f
            fsolfilename = get_fsol_filename(prob);
            prob.fsol = load("$(fsolfilename).jld", "fsol")
        catch
            println("No fsol for ", prob.name);
            prob.fsol = 0.0
        end
    else
        prob.fsol = 0.0
    end
end

function get_fsol_filename(prob)
    savename = string(replace(prob.name, r"[\/]" => "-"), "-fsol"); # julia 0.7
    default_path = "./data/";
    return string(default_path, savename);
end
