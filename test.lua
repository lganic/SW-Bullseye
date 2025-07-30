Real to astronomy:

function onTick()
	moonmode = input.getBool(1)
	x = input.getNumber(1)
	y = input.getNumber(2)
	z = input.getNumber(3)
	
	output.setNumber(3,y)
	
	if(moonmode) then
		output.setNumber(1,X(x+200000,y,z))
		output.setNumber(2,Y(x,y,z+80000))
	else
		output.setNumber(1,X(x,y,z))
		output.setNumber(2,Y(x,y,z))
	end
end
	
function X(a,b,c)
	if(c<=128000) then
		if(a<100000) then
			return a
		else
			return 200000-a
		end
	end
	return 100000-math.sqrt(((a-100000)^2)+((c-128000)^2))
end
	
function Y(a,b,c)
	if(c<=128000) then
		if(a<100000) then
			return c
		else
			return 490159.265359-c
		end
	end
	return 442159.265359-(100000*math.atan(c-128000, a-100000))
end