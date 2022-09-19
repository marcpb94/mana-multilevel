exec: ../lulesh2.0 
	mkdir -p local && mkdir -p global
	rm -rf local/* && rm -rf global/*
	../../../bin/mana_coordinator --daemon --config mana.conf
	mpirun -np 8 ../../../bin/mana_launch --config mana.conf ../lulesh2.0 -i 2400 -s 60 -p

restart: 
	../../../bin/mana_coordinator --daemon --config mana.conf
	mpirun -np 8 ../../../bin/mana_restart --config mana.conf

clean:
	rm -rf local/* && rm -rf global/*

