
#ifndef CONFIG_H
#define CONFIG_H

#include "../jalib/jassert.h"
#include "../jalib/jconvert.h"
#include <iostream>
#include <fstream>
#include <string>


#define GLOBAL_CKPT_DIR_OPTION "global_ckpt_dir"
#define LOCAL_CKPT_DIR_OPTION "local_ckpt_dir"
#define GLOBAL_CKPT_INT_OPTION "global_ckpt_interval"
#define PARTNER_CKPT_INT_OPTION "partner_ckpt_interval"
#define LOCAL_CKPT_INT_OPTION "local_ckpt_interval"
#define TEST_MODE_OPTION "test_mode"


struct ConfigInfo {

public:

  std::string globalCkptDir;
  std::string localCkptDir;
  uint32_t globalInterval;
  uint32_t partnerInterval;
  uint32_t localInterval;
  uint32_t testMode;

  ConfigInfo();
  void readConfigFromFile(std::string filename);
  static void writeRestartDir(std::string val);
  static std::string readRestartDir();
};

struct Topology {

public:

  int numNodes;
  char *nameList, *hostname;
  int *nodeMap, *partnerMap;

  Topology(int num_nodes, char *name_list, char *host_name, int *node_map, int *partner_map);

};



#endif //ifndef CONFIG_H
