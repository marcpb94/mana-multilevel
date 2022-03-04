#ifndef CONFIG_MPI_H
#define CONFIG_MPI_H

#include "../jalib/jassert.h"
#include "../jalib/jconvert.h"
#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include "util_config.h"
#include "constants.h"

using namespace dmtcp;

struct UtilsMPI {

private:

  int _rank = -1;
  int _size = -1;
  char hostName[HOSTNAME_MAXSIZE];

public:

  UtilsMPI();
  int getRank() const { return _rank; }
  int getSize() const { return _size; }
  char* getHostname(int test_mode);
  void getSystemTopology(ConfigInfo cfg);
  string recoverFromCrash(ConfigInfo cfg);
  static UtilsMPI instance();

};

#endif //ifndef CONFIG_MPI_H
