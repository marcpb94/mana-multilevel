all: heatdis-mpi heatdis-mana

heatdis-mpi: heatdis.c
	mpicc -o heatdis.mpi.exe heatdis.c -lm

heatdis-mana: heatdis.c
	gcc -o heatdis.mana.exe heatdis.c -I../../include/ -L../../lib/dmtcp -lm -lmpistub


exec:	heatdis-mana
	mkdir -p local && mkdir -p global
	rm -rf local/* && rm -rf global/*
	../../bin/mana_coordinator --daemon --config mana.conf && mpirun -np 4 ../../bin/mana_launch --config mana.conf ./heatdis.mana.exe 150

restart: heatdis-mana
	../../bin/mana_coordinator --daemon --config mana.conf && mpirun -np 4 ../../bin/mana_restart --config mana.conf

clean:
	rm -rf *.exe local/* global/*
