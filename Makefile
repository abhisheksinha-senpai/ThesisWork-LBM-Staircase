sim : 
	nvcc -arch sm_60 main.cu utilities.cu boundary.cu lbm.cu -o sim

clean :
	rm -rf sim