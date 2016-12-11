local ParentModule = torch.class('unsupervised.ParentModule')

function ParentModule:__init()
end

-- Mean Zero the data
function ParentModule:_meanZero(X, mean, inplace)
   local inplace = inplace or false

   -- Mean zero the X
   local normedX = X.new()
   if inplace then
      normedX = X
   else
      normedX:resizeAs(X):copy(X)
   end
   local tempMean = mean.new()
   tempMean:resizeAs(mean):copy(mean)
   tempMean:expandAs(tempMean, normedX)
   local expandedMean = mean.new()
   expandedMean:resizeAs(tempMean):copy(tempMean)
   normedX:csub(expandedMean)
   return normedX
end

-- function process does the unsupervised learning
function ParentModule:train()
   error('Every unsupervised algorithm implements this function.')
end
