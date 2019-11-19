function load_fsol!(options, prob  ;  data_path::AbstractString="./data/")
    if(options.exacterror)
        try # getting saved optimal f
            fsolfilename = get_fsol_filename(prob, data_path =data_path);
            prob.fsol = load("$(fsolfilename).jld", "fsol")
            print(fsolfilename)
            print(prob.fsol)
        catch
            println("No fsol for ", prob.name);
            prob.fsol = 0.0
        end
    else
        prob.fsol = 0.0
    end
end

function get_fsol_filename(prob ;  data_path::AbstractString="./data/")
    savename = string(replace(prob.name, r"[\/]" => "-"), "-fsol");
#     data_path = "./data/";
    return string(data_path, savename);
end