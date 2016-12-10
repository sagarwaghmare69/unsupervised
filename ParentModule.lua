local ParentModule = torch.class('unsupervised.ParentModule')

function ParentModule:__init()
end

-- function process does the unsupervised learning
function ParentModule:process()
   error('Every unsupervised algorithm implements this function.')
end
