using PyPlot

function plot2Dclassifier(X,y,model;Xtest=[],ytest=[],biasIncluded=false,k=2)

	if biasIncluded
		f1 = 2
		f2 = 3
	else
		f1 = 1
		f2 = 2
	end

	increment = 100

	# Pick some colors for the classes (for binary it uses red/blue)
	colours = [0 1 0
			1 0 0
			0 0 1
			1 1 1
			1 0 1
			0 1 1
			1 1 0
			.1 .1 .1
			1 .5 0
			0 .5 0
			.5 .5 .5
			.5 .25 0
			.5 0 .5
			0 .5 1]
	figure()
	if k == 2
		plot(X[y.==1,f1],X[y.==1,f2],"b+")
		plot(X[y.==2,f1],X[y.==2,f2],"ro")
		if !isempty(Xtest)
			plot(Xtest[ytest.==1,f1],Xtest[ytest.==1,f2],"bx");
			plot(Xtest[ytest.==2,f1],Xtest[ytest.==2,f2],"rs");
		end
	else

		for c in 1:k
			plot(X[y.==c,f1],X[y.==c,f2],"+",color=(colours[c,1],colours[c,2],colours[c,3]))
			if !isempty(Xtest)
				plot(Xtest[ytest.==c,f1],Xtest[ytest.==c,f2],"x",color=(colours[c,1],colours[c,2],colours[c,3]))
			end
		end
	end

	(xmin,xmax) = xlim()
	xDomain = range(xmin,stop=xmax,length=increment)
	(ymin,ymax) = ylim()
	yDomain = range(ymin,stop=ymax,length=increment)

	xValues = repeat(xDomain,1,length(xDomain))
	yValues = repeat(yDomain',length(yDomain),1)

	if biasIncluded
		t = length(xValues)
		z = model.predict([ones(t,1) xValues[:] yValues[:]])
	else
		z = model.predict([xValues[:] yValues[:]])
	end
	@assert(length(z) == length(xValues),"Size of model function's output is wrong");

	zValues = reshape(z,size(xValues))

	if k == 2
		if all(zValues[:] == 1)
    			cm = [0 0 .5];
		elseif all(zValues[:] == 2)
    			cm = [.5 0 0];
		else
	    		cm = [0 0 .5;.5 0 0];
		end

		contourf(xValues,yValues,zValues)
	else
		cm = zeros(0,3)
		nColours = 0
		for c in 1:k
			if any(zValues[:] .== c)
				nColours += 1
				cm = [cm;.5*colours[c,:]']
				zValues[zValues .== c] .= nColours
			end
		end
		levels = 0:nColours
		contourf(xValues,yValues,zValues,levels=levels,cmap=ColorMap("myColourMap",cm))
	end
end
