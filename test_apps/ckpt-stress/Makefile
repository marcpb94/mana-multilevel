all: ckpt-mpi ckpt-mana

ckpt-mpi: ckpt-stress.c
	mpicc -o ckpt.mpi.exe ckpt-stress.c -lm

ckpt-mana: ckpt-stress.c
	gcc -o ckpt.mana.exe ckpt-stress.c -I../../include/ -L../../lib/dmtcp -lm -lmpistub


exec:	ckpt-mana
	mkdir -p local && mkdir -p global
	rm -rf local/* && rm -rf global/*
	../../bin/mana_coordinator --daemon --config mana.conf && mpirun -np 4 ../../bin/mana_launch --config mana.conf ./ckpt.mana.exe 800 600

restart: ckpt-mana
	../../bin/mana_coordinator --daemon --config mana.conf && mpirun -np 4 ../../bin/mana_restart --config mana.conf

clean:
	rm -rf *.exe local/* global/*
