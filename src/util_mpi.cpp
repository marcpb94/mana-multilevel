#include "util_mpi.h"

static UtilsMPI *_inst = NULL;

UtilsMPI::UtilsMPI(){
  int init;
  MPI_Initialized(&init);
  if(!init){
    MPI_Init(NULL, NULL);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
}

UtilsMPI
UtilsMPI::instance(){
  if (_inst == NULL){
    _inst = new UtilsMPI();
  }
  return *_inst;
}

void
UtilsMPI::getSystemTopology(int test_mode, Topology **topo)
{
  char *hostname;
  int num_nodes;
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  hostname = UtilsMPI::instance().getHostname(test_mode);
  num_nodes = 0;
  //allocate enough memory for all hostnames
  char *allNodes = (char *)malloc(HOSTNAME_MAXSIZE * mpi_size);
  char *nameList = (char *)malloc(HOSTNAME_MAXSIZE * mpi_size);
  int *nodeMap = (int *)malloc(sizeof(int) * mpi_size);
  int *partnerMap = (int *)malloc(sizeof(int) * mpi_size);
  memset(nameList, 0, HOSTNAME_MAXSIZE * mpi_size);
  memset(nodeMap, 0, sizeof(int) * mpi_size);
  //distribute all hostnames
  MPI_Allgather(hostname, HOSTNAME_MAXSIZE, MPI_CHAR, allNodes,
                  HOSTNAME_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);

  //create rank-node mapping
  num_nodes = 0;
  int i, j, found;
  for (i = 0; i < mpi_size; i++){
    found = 0;
    //check if already in the list
    for (j = 0; j < num_nodes && !found; j++){
      if(strcmp(nameList + j*HOSTNAME_MAXSIZE, allNodes + i*HOSTNAME_MAXSIZE) == 0){
        found = 1;
        break;
      }
    }
    if (!found){
      //add node mapping
      nodeMap[i] = num_nodes;
      //add new node to nodelist
      strcpy(nameList + num_nodes*HOSTNAME_MAXSIZE, allNodes + i*HOSTNAME_MAXSIZE);
      num_nodes++;
    }
    else {
      //add node mapping
      nodeMap[i] = j;
    }
  }

  //initialize partner map with impossible value
  for(i = 0; i < mpi_size; i++) partnerMap[i] = -1;
  //create partner mapping
  for (i = 0; i < mpi_size; i++){
    if(partnerMap[i] == -1){
      found = 0;
      //find available partner rank in different node
      for(j = i+1; j < mpi_size; j++){
        if(nodeMap[i] != nodeMap[j] && partnerMap[j] == -1){
          //connect both nodes
          partnerMap[i] = j;
          partnerMap[j] = i;
          found = 1;
          break;
        }
      }
      JASSERT(found).Text("Could not complete partner mapping, application topology might contain uneven ranks per node or the execution is performed locally without enabling test mode.");
    }
  }

  *topo = new Topology(num_nodes, nameList, hostname, nodeMap, partnerMap);

  free(allNodes);
}

char *
UtilsMPI::getHostname(int test_mode)
{
  char *hostName = (char *)malloc(HOSTNAME_MAXSIZE);
  if(!test_mode){
    JASSERT(gethostname(hostName, sizeof hostName) == 0) (JASSERT_ERRNO);
  }
  else {
    //fake node size for testing purposes
    int node_size = 2;
    sprintf(hostName, "node%d", _rank/node_size);
  }
  return hostName;
}

void
UtilsMPI::performPartnerCopy(string ckptFilename, int *partnerMap){
  printf("Starting to perform partner copy...\n");
  fflush(stdout);
  
  string partnerFilename = ckptFilename + "_partner";
  string partnerChksum = partnerFilename + "_md5chksum";
  string ckptChksum = ckptFilename + "_md5chksum";
  int myPartner = partnerMap[_rank];
  
  int fd_p = open(partnerFilename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_p != -1);

  int fd_m = open(ckptFilename.c_str(), O_RDONLY);
  JASSERT(fd_m != -1);
  
  int fd_p_chksum = open(partnerChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_p_chksum != -1);

  int fd_m_chksum = open(ckptChksum.c_str(), O_RDONLY);
  JASSERT(fd_m_chksum != -1);

  struct stat sb;
  JASSERT(fstat(fd_m, &sb) == 0);

  off_t ckptSize = sb.st_size, partnerCkptSize;
  off_t toSend = ckptSize, toRecv = 0;
  char *buff = (char *)malloc(DATA_BLOCK_SIZE);

  //decide send/recv order
  if(_rank > myPartner){
    //exchange ckpt file sizes
    MPI_Send(&ckptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
    MPI_Recv(&partnerCkptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    toRecv = partnerCkptSize;
    //send ckpt file
    while (toSend > 0){
      off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toSend;

      Util::readAll(fd_m, buff, sendSize);
      MPI_Send(buff, sendSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
      toSend -= sendSize;
    }
    //receive partner copy
    while(toRecv > 0){
      off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toRecv;

      MPI_Recv(buff, recvSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      Util::writeAll(fd_p, buff, recvSize);
      toRecv -= recvSize;
    }
    //send checksum
    Util::readAll(fd_m_chksum, buff, 16);
    MPI_Send(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
    //receive checksum
    MPI_Recv(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Util::writeAll(fd_p_chksum, buff, 16);
  }
  else {
    //exchange ckpt file sizes
    MPI_Recv(&partnerCkptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    toRecv = partnerCkptSize;
    MPI_Send(&ckptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
    //receive partner copy
    while(toRecv > 0){
      off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toRecv;

      MPI_Recv(buff, recvSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      Util::writeAll(fd_p, buff, recvSize);
      toRecv -= recvSize;
    }
    //send ckpt file
    while (toSend > 0){
      off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toSend;

      Util::readAll(fd_m, buff, sendSize);
      MPI_Send(buff, sendSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
      toSend -= sendSize;
    }
    //receive checksum
    MPI_Recv(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Util::writeAll(fd_p_chksum, buff, 16);
    //send checksum
    Util::readAll(fd_m_chksum, buff, 16);
    MPI_Send(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
  }

  printf("Finished performing partner copy.\n");
  fflush(stdout);

  free(buff);
  JASSERT(close(fd_p) == 0);
  JASSERT(close(fd_m) == 0);
  JASSERT(close(fd_p_chksum) == 0);
  JASSERT(close(fd_m_chksum) == 0);
}

const char*
findCkptFilename(string ckpt_dir, string patternEnd){
  int file_found = 0;  
  DIR *dir;
  struct dirent *entry;
  string result = ckpt_dir;

  dir = opendir(ckpt_dir.c_str());
  if (dir == NULL) {
    printf("  Checkpoint directory %s not found.\n", ckpt_dir.c_str());
    return 0;
  }
  while ((entry = readdir(dir)) != NULL){
    if(Util::strStartsWith(entry->d_name, "ckpt") &&
            Util::strEndsWith(entry->d_name, patternEnd.c_str())){
      result.append("/").append(entry->d_name);
      file_found = 1;
      break;
    }
  }
  return (file_found) ? result.c_str() : NULL;
}


/**
 *
 * Fake reading of ProcessInfo data for accessing later data
 * to avoid modifying the underlying process and accidentally
 * open the gates of hell.
 *
 */
void
dummySerialize(jalib::JBinarySerializer &o){
  uint64_t dummy64;
  uint32_t dummy32;
  pid_t dummyPid;
  UniquePid dummyUPid;
  string dummyStr;
  map<pid_t, UniquePid> dummyMap;

  JSERIALIZE_ASSERT_POINT("ProcessInfo:");

  o & dummy32;
  o & dummy32 & dummyPid & dummyPid & dummyPid & dummyPid & dummyPid & dummy32;
  o & dummyStr & dummyStr & dummyStr & dummyStr & dummyStr;
  o & dummyUPid & dummyUPid;
  o & dummy64 & dummy64
    & dummy64 & dummy64;
  o & dummyUPid & dummy32 & dummy32 & dummy32 & dummy32;
  o & dummy64 & dummy64 & dummy64;
  o & dummy64 & dummy64 & dummy64 & dummy64 & dummy64;
  int i;
  for (i = 0; i <= CKPT_GLOBAL; i++){
    o & dummyStr & dummyStr & dummyStr;
  }
  o & dummyMap;

  JSERIALIZE_ASSERT_POINT("EOF");
}

int
readDmtcpHeader(int fd){
  const size_t len = strlen(DMTCP_FILE_HEADER);
  
  char *buff = (char *)malloc(len);
  if(Util::readAll(fd, buff, len) != len){
    free(buff);
    return 0;
  }
  free(buff);

  jalib::JBinarySerializeReaderRaw rdr("", fd);

  //careful now....
  dummySerialize(rdr);

  size_t numRead = len + rdr.bytes();
  
  // We must read in multiple of PAGE_SIZE
  const ssize_t pagesize = Util::pageSize();
  ssize_t remaining = pagesize - (numRead % pagesize);
  char buf[remaining];

  return (Util::readAll(fd, buf, remaining) == remaining);
}

int
readMtcpHeader(int fd){
  MtcpHeader mtcpHdr;
  return (Util::readAll(fd, &mtcpHdr, sizeof(mtcpHdr)) == sizeof(mtcpHdr)); 
}

int
UtilsMPI::isCkptValid(const char *filename){
  MD5_CTX context;
  unsigned char digest[16], chksum_read[16];
  string ckptFile = filename;
  string ckptChecksum = ckptFile + "_md5chksum";

  int fd = open(ckptFile.c_str(), O_RDONLY);
  int fd_chksum = open(ckptChecksum.c_str(), O_RDONLY);
  
  if (fd == -1 || fd_chksum == -1){
    //treat the absence or lack of ablity to open
    //either of the two files as a failure
    printf("  Checkpoint or checksum file missing.\n");
    if(fd != -1) close(fd);
    if(fd_chksum != -1) close(fd_chksum);
    return 0;
  }

  //read checksum file
  if(Util::readAll(fd_chksum, chksum_read, 16) != 16){
    printf("  Checksum file is smaller than 16 bytes.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  //ignore DMTCP header
  if(!readDmtcpHeader(fd)){
    printf("  Error reading DMTCP header.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  //ignore MTCP header
  if(!readMtcpHeader(fd)){
    printf("  Error reading MTCP header.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  //read checkpoint file as pure data just to verify checksum
  Area area;
  size_t END_OF_CKPT = -1;
  ssize_t bytes_read;
  int skip_update;
  int num_updates = 0;
  MD5_Init(&context);
  while (Util::readAll(fd, &area, sizeof(area)) == sizeof(area)){
    skip_update = 0;

    //if -1 size found, we are finished
    if (area.size == END_OF_CKPT) break;

    //printf("Reading region of %lu bytes...\n", area.size);
    //fflush(stdout);

    //printf("area name: %s\n", area.name);


    //handle non rwx and anonymous pages
    if (area.prot == 0 || (area.name[0] == '\0' &&
          ((area.flags & MAP_ANONYMOUS) != 0) &&
          ((area.flags & MAP_PRIVATE) != 0))){
      if (area.properties == 0){
        //some contents have been written
        //read but ignore them
        skip_update = 1;
      }
      else {
        //zero page, skip
        continue;
      }
    }

    
    if ((area.properties & DMTCP_SKIP_WRITING_TEXT_SEGMENTS) &&
          (area.prot & PROT_EXEC)){
       //skip text segment if applicable
       continue;
    }

    //read memory region
    area.addr = (char *)malloc(area.size);
    if((bytes_read = Util::readAll(fd, area.addr, area.size)) != (ssize_t)area.size){
      if(bytes_read == -1) {
        printf("  Error reading memory region: %s\n", strerror(errno));
      }
      else {
        printf("  Expected memory region of %lu bytes, got only %ld.\n", area.size, bytes_read);
      }
      close(fd); close (fd_chksum);
      free(area.addr);
      return 0;
    }

    if(!skip_update){
      //update md5 context
      MD5_Update(&context, area.addr, area.size);

      //testing
      printf("Rank: %d, region name:%s, checksum(%d): ", _rank, area.name, num_updates);
      MD5_Final(digest, &context);
      for(int i = 0; i < 16; i++){
        printf("%x", digest[i]);
      }
      printf("\n");
      num_updates++;
      
      //TEST, REMOVE AFTERWARDS
      MD5_Init(&context);
    }

    free(area.addr);
  }

  if(area.size != END_OF_CKPT){
    printf("  Checkpoint file did not finish as expected.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  MD5_Final(digest, &context);

  printf("Number of updates: %d\n", num_updates);
  printf("Computed checksum: ");
  for(int i = 0; i < 16; i++){
    printf("%x", digest[i]);
  }
  printf("\nRead checksum: ");
  for(int i = 0; i < 16; i++){
    printf("%x", chksum_read[i]);
  }
  printf("\n");
 
  if(strncmp((char *)digest, (char *)chksum_read, 16) != 0){
    printf("  Computed checksum does not match the checksum file contents.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  close(fd);
  close(fd_chksum);
  return 1;
}


/**
 *
 * Asserting anything in this function and its 
 * subfunctions is a bad idea, since crashing the
 * application here defeats its purpose, as we 
 * want to choose alternative checkpoint levels 
 * if the the checkpoint being checked has any 
 * issues.
 *
 */
int
UtilsMPI::checkCkptValid(int ckpt_type, string ckpt_dir){
  string real_dir = ckpt_dir;
  char *rankchar = (char *)malloc(32);
  sprintf(rankchar, "/ckpt_rank_%d/", _rank);
  real_dir += rankchar;
  int success = 0, allsuccess;  
  string ckptFilename;

  try {
    const char *filename = findCkptFilename(real_dir, ".dmtcp");
    if(filename != NULL && isCkptValid(filename)){
        success = 1;
    }
    else {
      if(ckpt_type == CKPT_PARTNER){
        //TODO: also recover from partner if possible
        
      }
    }
  }
  catch (...){} //survive unexpected exceptions and assume failure

  printf("Success of rank %d: %d\n", _rank, success);
  fflush(stdout);

  //aggregate operation status from all ranks
  MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  free(rankchar);

  // only return success if all ranks are successful
  return allsuccess;
}

string
UtilsMPI::recoverFromCrash(ConfigInfo *cfg){
    RestartInfo *restartInfo = new RestartInfo();
    restartInfo->readRestartInfo();
    string target = "";
    int i, found = 1, valid = 0, candidate;

    /*
    for(int i = 0; i <= CKPT_GLOBAL; i++){
      printf("type: %d, dir: %s, time: %ld\n", i, restartInfo->ckptDir[i].c_str(), restartInfo->ckptTime[i]);
    }
    fflush(stdout);
    */

    //recovery prioritizing by time of checkpoint
    while(found && !valid){
      found = 0;
      candidate = 0;
      for(i = 0; i <= CKPT_GLOBAL; i++){
        if(restartInfo->ckptTime[i] > 0 && (!found || 
              restartInfo->ckptTime[i] > restartInfo->ckptTime[candidate])){
        
          //update candidate
          candidate = i;
          found = 1;
        }
      }
      if(found){
        //check if the checkpoint is actually valid
        target = restartInfo->ckptDir[candidate];
        printf("Checking checkpoint dir %s...\n", target.c_str());
        valid = checkCkptValid(candidate, target);
        if(!valid){
          //if not valid, set time to zero to invalidate
          //and loop again
          restartInfo->ckptTime[candidate] = 0;
        }
      }
    }

    JASSERT(found).Text("Restart point not found.");

    printf("Selected checkpoint type %d with location %s\n", candidate, target.c_str());
    fflush(stdout);

    return target;
}

