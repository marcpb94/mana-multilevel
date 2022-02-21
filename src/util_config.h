
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
#define LOCAL_CKPT_INT_OPTION "local_ckpt_interval"

struct ConfigInfo {
  std::string globalCkptDir;
  std::string localCkptDir;
  uint32_t globalInterval;
  uint32_t localInterval;

  ConfigInfo();
  void readConfigFromFile(std::string filename);
  static void writeRestartDir(std::string val);
  static std::string readRestartDir();
};


#endif //ifndef CONFIG_H
