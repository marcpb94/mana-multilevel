#include "util_mpi.h"
#include "galois.h"


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
UtilsMPI::getSystemTopology(ConfigInfo *cfg, Topology **topo)
{
  char *hostname;
  int num_nodes;
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  hostname = getHostName(cfg);
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

  int node_size = mpi_size / num_nodes;

  if (node_size != cfg->nodeSize && _rank == 0){
    printf("Real node size(%d) does not match specified value(%d), is this intended?\n", node_size, cfg->nodeSize);
    fflush(stdout);
  }

  // We first want to create an inverse node-process mapping as that done in Topology. There, nodeMap[i] linked
  // the ith process to the ith node in nameList. Instead, we want that in inverseNodeMap the node is indicated 
  // by the position in the array (e.g. the first n processes correspond to the first node, the next n processes
  // correspond to the second one, etc; where n is the number of processes per node). Then, inverseNodeMap[i] 
  // links the ith process to the node corresponding to its position in the array. This will the grouping of nodes
  // easier.

  int num_proc = mpi_size;

  int* inverseNodeMap = (int *)malloc(sizeof(int) * num_proc);

  // We initally set the mapping incorrectly
  for(i=0; i<num_proc; i++){
    inverseNodeMap[i] = -1;
  }

  for(i=0; i<num_proc; i++){
    int pos = nodeMap[i]*node_size;
    for(j=0;j<node_size;j++){
      // We search for an unoccupied space in the processes corresponding to a given node
      if(inverseNodeMap[pos+j]==-1){
        // inverseNodeMap[pos+j] is not occupied so we put process i here
        inverseNodeMap[pos+j]=i;
        break;
      }
    }
  }

  // Now all processes are nicely organized in inverseNodeMap depending on the node to which they belong. 
  // We want now to group different processes correponding to different nodes with a size set by group_size
  // and create communicators around them.
  int group_size = cfg->groupSize;
  MPI_Group newGroup, origGroup;
  MPI_Comm group_comm;
  int group_rank;
  int right, left;
  int my_node = nodeMap[mpi_rank];
  int section_ID = my_node / group_size;
  int buf = section_ID*group_size*node_size;
  int group[group_size];

  // Make sure that group size is multiple of the node size
  JASSERT(num_nodes % group_size == 0)(num_nodes)(group_size).Text("The number of nodes must be multiple of the group size.");

  // We have to find where our process is located in inverseNodeMap. We already now it corresponds to node nodeMap[mpi_rank]
  int pos;
  for(pos = 0;pos<node_size;pos++){
    if(inverseNodeMap[nodeMap[mpi_rank]*node_size+pos]==mpi_rank){
      break;
    }
  }

  for(i = 0;i<group_size;i++){
    group[i] = inverseNodeMap[buf+pos+i*node_size];
  }

  MPI_Comm_group(MPI_COMM_WORLD, &origGroup);
  MPI_Group_incl(origGroup, group_size, group, &newGroup);
  MPI_Comm_create(MPI_COMM_WORLD, newGroup, &group_comm); 
  MPI_Comm_rank(group_comm, &group_rank);
  right=(group_rank+1+group_size) % group_size;
  left=(group_rank-1+group_size) % group_size;


  MPI_Group_free(&origGroup);
  MPI_Group_free(&newGroup);

  //initialize partner map with impossible value
  for(i = 0; i < mpi_size; i++) partnerMap[i] = -1;
  //create partner mapping
  for (i = 0; i < mpi_size; i++){
    // The process is at node nodeMap[i]. Let us search at which position in inverseNodeMap.
    int buf = nodeMap[i]*node_size;
    int j;
    for(j=0;j<node_size;j++){
      if(i==inverseNodeMap[buf+j]){
        break;
      }
    }
    int sect = nodeMap[i] / group_size;
    partnerMap[i]= inverseNodeMap[sect*group_size*node_size + (buf+j+node_size)%(group_size*node_size)];
  }


  *topo = new Topology(num_nodes, nameList, hostname, nodeMap, partnerMap, mpi_size, node_size, group_size, section_ID, group_rank, right,
                      left, group_comm);

  free(allNodes);
}

char *
UtilsMPI::getHostName(ConfigInfo *cfg)
{
  char *hostName = (char *)malloc(HOSTNAME_MAXSIZE);
  if(!cfg->testMode){
    JASSERT(gethostname(hostName, HOSTNAME_MAXSIZE) == 0) (JASSERT_ERRNO);
  }
  else {
    //fake node size for testing purposes
    int node_size = cfg->nodeSize;
    sprintf(hostName, "node%d", _rank/node_size);
  }
  return hostName;
}

int
UtilsMPI::assistPartnerCopy(string ckptFilename, Topology *topo){
  
  printf("Rank %d | Group rank %d: performing recovery for partner group rank %d...\n", _rank, topo->groupRank, topo->left);
  fflush(stdout);
  
  string partnerCkptFile = ckptFilename;
  string partnerChksum = ckptFilename + "_md5chksum";
 
  int fd = open(partnerCkptFile.c_str(), O_RDONLY);
  if(fd == -1) return 0;
  
  int fd_chksum = open(partnerChksum.c_str(), O_RDONLY);
  if(fd_chksum == -1) return 0;

  struct stat sb;
  if(fstat(fd, &sb) != 0) return 0;
  
  off_t ckptSize = sb.st_size;
  off_t toSend = ckptSize;
  char *buff = (char *)malloc(DATA_BLOCK_SIZE);

  //exchange ckpt file sizes
  MPI_Send(&ckptSize, sizeof(off_t), MPI_CHAR, topo->left, 0, topo->groupComm);
  //send ckpt file
  while(toSend > 0){
    off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
			  DATA_BLOCK_SIZE : toSend;

    Util::readAll(fd, buff, sendSize);
    MPI_Send(buff, sendSize, MPI_CHAR, topo->left, 0, topo->groupComm);
    toSend -= sendSize;
  }
  //send checksum
  Util::readAll(fd_chksum, buff, 16);
  MPI_Send(buff, 16, MPI_CHAR, topo->left, 0, topo->groupComm);
 


  free(buff);
  close(fd);
  close(fd_chksum);

  return 1;
}

int
UtilsMPI::recoverFromPartnerCopy(string ckptFilename, Topology *topo){
  
  printf("Rank %d | Group rank %d: recovering from partner group rank %d...\n", _rank, topo->groupRank, topo->right);
  fflush(stdout);
  
  string ckptChksum = ckptFilename + "_md5chksum";
  
  int fd = open(ckptFilename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  if(fd == -1) return 0;
  
  int fd_chksum = open(ckptChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  if(fd_chksum == -1) return 0;

  off_t ckptSize;
  off_t toRecv;
  char *buff = (char *)malloc(DATA_BLOCK_SIZE);

  //exchange ckpt file sizes
  MPI_Recv(&ckptSize, sizeof(off_t), MPI_CHAR, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
  toRecv = ckptSize;
  //receive ckpt file
  while(toRecv > 0){
    off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
			  DATA_BLOCK_SIZE : toRecv;

    MPI_Recv(buff, recvSize, MPI_CHAR, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
    Util::writeAll(fd, buff, recvSize);
    toRecv -= recvSize;
  }
  //receive checksum
  MPI_Recv(buff, 16, MPI_CHAR, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
  Util::writeAll(fd_chksum, buff, 16);


  free(buff);
  close(fd);
  close(fd_chksum);
 
  return 1;
}

void
UtilsMPI::performPartnerCopy(string ckptFilename, Topology *topo){

  if (_rank == 0){
    printf("Performing partner copy...\n");
    fflush(stdout);
  }
  
  string partnerFilename = ckptFilename + "_partner";
  string partnerChksum = partnerFilename + "_md5chksum";
  string ckptChksum = ckptFilename + "_md5chksum";
  
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
  char *buffRecv = (char *)malloc(DATA_BLOCK_SIZE);
  char *buffSend = (char *)malloc(DATA_BLOCK_SIZE);
  MPI_Request req;

  //identify left and right partners
  int group_rank, group_size;
  int right, left;
  right = topo->right;
  left = topo->left;

  //exchange ckpt file sizes
  MPI_Isend(&ckptSize, sizeof(off_t), MPI_CHAR, right, 0, topo->groupComm, &req);
  MPI_Recv(&partnerCkptSize, sizeof(off_t), MPI_CHAR, left, 0, topo->groupComm, MPI_STATUS_IGNORE);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  toRecv = partnerCkptSize;
  //send/recv ckpt file
  while (toSend > 0 || toRecv > 0){
    off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
                         DATA_BLOCK_SIZE : toSend;

    off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
                         DATA_BLOCK_SIZE : toRecv;

    if (sendSize > 0) {
      Util::readAll(fd_m, buffSend, sendSize);
      MPI_Isend(buffSend, sendSize, MPI_CHAR, right, 0, topo->groupComm, &req);
    }
    if(recvSize > 0){
      MPI_Recv(buffRecv, recvSize, MPI_CHAR, left, 0, topo->groupComm, MPI_STATUS_IGNORE);
      Util::writeAll(fd_p, buffRecv, recvSize);
    }
    if(sendSize > 0){
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
      
    toSend -= sendSize;
    toRecv -= recvSize;
  }

  Util::readAll(fd_m_chksum, buffSend, 16);
  MPI_Isend(buffSend, 16, MPI_CHAR, right, 0, topo->groupComm, &req);
  MPI_Recv(buffRecv, 16, MPI_CHAR, left, 0, topo->groupComm, MPI_STATUS_IGNORE);
  Util::writeAll(fd_p_chksum, buffRecv, 16);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  if (_rank == 0){
    printf("Finished partner copy.\n");
    fflush(stdout);
  }

  free(buffSend);
  free(buffRecv);
  JASSERT(close(fd_p) == 0);
  JASSERT(close(fd_m) == 0);
  JASSERT(close(fd_p_chksum) == 0);
  JASSERT(close(fd_m_chksum) == 0);
}

void 
UtilsMPI::performRSEncoding_w16(string ckptFilename, Topology* topo){

  int w = 8;
  // Group rank 0 is the process that will perform RS encoding from the blocks of ckpt image the other processes send him.
  // After each block of encoded ckpt file has been calculated, group rank 0 sends to the other processes their corresponding part.

  if(topo->groupRank==0){
    printf("Performing RS encoding...\n");
    fflush(stdout);
  }

  string encodedFilename = ckptFilename + "_encoded";
  string encodedChksum = ckptFilename + "_encoded_md5chksum";

  int fd_m = open(ckptFilename.c_str(), O_RDONLY);
  JASSERT(fd_m != -1);

  int fd_e = open(encodedFilename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_e != -1);

  struct stat sb;
  JASSERT(fstat(fd_m, &sb) == 0);
  off_t myCkptFileSize = sb.st_size;

  int i;
  off_t ckptFileSizes[topo->groupSize];
  if(topo->groupRank==0){
    ckptFileSizes[0]=myCkptFileSize;
    for(i=1;i<topo->groupSize;i++){
      MPI_Recv(&(ckptFileSizes[i]),sizeof(off_t),MPI_CHAR,i,0,topo->groupComm,MPI_STATUS_IGNORE);
    }
  }else{
    MPI_Send(&myCkptFileSize,sizeof(off_t),MPI_CHAR,0,0,topo->groupComm);
  }
  off_t maxSize;
  if(topo->groupRank==0){
    maxSize = ckptFileSizes[0];
    for(i=1;i<topo->groupSize;i++){
      if(ckptFileSizes[i]>maxSize){
        maxSize=ckptFileSizes[i];
      }
    }
    for(i=1;i<topo->groupSize;i++){
      MPI_Send(&maxSize,sizeof(off_t),MPI_CHAR,i,0,topo->groupComm);
    }
  }else{
    MPI_Recv(&maxSize,sizeof(off_t),MPI_CHAR,0,0,topo->groupComm,MPI_STATUS_IGNORE);
  }
  printf("Rank %d has ckptFileSize %lu\n",topo->groupRank,myCkptFileSize);
  fflush(stdout);

  // Now we have to elongate every ckpt file so that everyone has the same size. Their size is going to be maxSize+sizeof(off_t). 
  // Except that the size of each file has to be a multiple of w*machine_word_size, so we also have to round in this way

  off_t final_size = maxSize+sizeof(off_t);

  JASSERT(close(fd_m) == 0);
  if(truncate(ckptFilename.c_str(),final_size) == -1){
    JASSERT(0).Text("Error with truncation on ckpt image.\n");
  }
  
  // Now we will write the original size of the ckpt to the end of the elongated file, so that at restart we can recover 
  // the original ckpt image

  fd_m = open(ckptFilename.c_str(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_m != -1);
  if(lseek(fd_m, -sizeof(off_t), SEEK_END) == -1){
    JASSERT(0).Text("Unable to seek in file.\n");
  }
  if(write(fd_m,&myCkptFileSize,sizeof(off_t)) == -1){
    JASSERT(0).Text("Unable to write ckpt file size in elongated file.\n");
  }
  
  // Now let us turn to the actual encoding. 

  // We initialize the matrix. For now we will take a Cauchy matrix with X the first topo->groupSize elements of GF(2^w) and
  // Y the next topo->groupSize elements of GF(2^w). We build the matrix using GF(2^w).

  
  int *matrix;
  if(topo->groupRank==0){
    matrix = (int *)malloc(topo->groupSize*topo->groupSize*sizeof(int));
    int j;
    int X[topo->groupSize], Y[topo->groupSize];
    for(j=0;j<topo->groupSize;j++){
      X[j]=j;
    }
    for(j=topo->groupSize;j<2*topo->groupSize;j++){
      Y[j%topo->groupSize]=j;
    }
    for(i=0;i<topo->groupSize;i++){
      for(j=0;j<topo->groupSize;j++){
        matrix[i*topo->groupSize+j]=galois_single_divide(1,(X[i]^Y[j]),w);
      }
    }
  }
  
  int blockSize = 1024*1024; 
  
  fd_m = open(ckptFilename.c_str(), O_RDONLY);
  JASSERT(fd_m != -1);
  
  char *encodedBlocks; // Group rank 0 is going to temporarily store all the pieces of encoded files before sending them to 
                       // respective processes
  char *dataBlock; // All processes will need to store temporarily a piece of raw ckpt image.
  char *encodedBlock; // Every node is going to keep receiving from group rank 0 pieces of the final encoded files
  
  if(topo->groupRank==0){
    encodedBlocks = (char*)malloc(blockSize*topo->groupSize);
    encodedBlock = (char*)malloc(blockSize);
    dataBlock = (char*)malloc(blockSize);
  }else{
    encodedBlock = (char*)malloc(blockSize);
    dataBlock = (char*)malloc(blockSize);
  }

  int pos = 0;
  while(pos<final_size){
    if(final_size-pos<blockSize){
      blockSize=final_size-pos;
    }
    // We are going to generate topo->groupSize encoded ckpt images
    
    // We first deal with the part of the ckpt image we already have at groupRank 0
    
    if(topo->groupRank==0){
      Util::readAll(fd_m,dataBlock,blockSize);
      for(i=0;i<topo->groupSize;i++){
        int j;
        int matrix_element = matrix[i*topo->groupSize];
        for(j=0;j<blockSize;j++){
          encodedBlocks[i*blockSize+j]=galois_single_multiply(matrix_element,dataBlock[j], w);
          //TODO instead of using single operation, use region
        }
      }
    }

    // Now let us deal with the part of the ckpt images from the other processes
    if(topo->groupRank==0){
      int k;
      for(k=1;k<topo->groupSize;k++){
        // We need to retrieve a block of raw ckpt image from process k
        MPI_Recv(dataBlock,blockSize,MPI_CHAR,k,0,topo->groupComm,MPI_STATUS_IGNORE);

        for(i=0;i<topo->groupSize;i++){
          int j;
          int matrix_element = matrix[i*topo->groupSize+k];
          for(j=0;j<blockSize;j++){
            encodedBlocks[i*blockSize+j]= (encodedBlocks[i*blockSize+j]^galois_single_multiply(matrix_element,dataBlock[j], w));
            //TODO instead of using single operation, use region
          }
        } 
      }
    }else{
      Util::readAll(fd_m,dataBlock,blockSize);
      MPI_Send(dataBlock,blockSize,MPI_CHAR,0,0,topo->groupComm);
    }

    // Now we have to retrieve and store in file the encoded ckpt images
    MPI_Scatter(encodedBlocks,blockSize,MPI_CHAR,encodedBlock,blockSize,MPI_CHAR,0,topo->groupComm);
    Util::writeAll(fd_e,encodedBlock,blockSize);

    pos +=blockSize;
  }
  close(fd_e);

  if(topo->groupRank==0){
    free(matrix);
    free(encodedBlocks);
    free(dataBlock);
  }else{
    free(encodedBlock);
  }

  //read checkpoint file as pure data just to compute checksum
  #pragma region chksum
  MD5_CTX context;
  unsigned char digest[16];
  Area area;
  size_t END_OF_CKPT = -1;
  ssize_t bytes_read;
  int chunkSize = MEM_TMP_SIZE;
  char *addr_tmp = (char *)malloc(MEM_TMP_SIZE);
  MD5_Init(&context);


  fd_e = open(encodedFilename.c_str(), O_RDONLY);
  JASSERT(fd_e != -1);

  int fd_e_chksum = open(encodedChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_e_chksum != -1);


  pos = 0;
  while(pos<final_size){
    if(final_size-pos<chunkSize){
      chunkSize=final_size-pos;
    }
    if((bytes_read = Util::readAll(fd_e, addr_tmp, chunkSize)) != (ssize_t)chunkSize){
        if(bytes_read == -1) {
          printf("Error reading memory region: %s\n", strerror(errno));
          fflush(stdout);
        }
        else {
          printf("Expected memory region of %lu bytes, got only %ld.\n", chunkSize, bytes_read);
          fflush(stdout);
        }
        close(fd_e); close (fd_e_chksum);
        free(addr_tmp);
        return;
    }
    MD5_Update(&context,addr_tmp,chunkSize);
    pos += chunkSize;
  }
  MD5_Final(digest, &context);
  Util::writeAll(fd_e_chksum, digest, 16);
  
  close(fd_e);
  close(fd_e_chksum);
}

void UtilsMPI::performRSDecoding(string filename,Topology* topo, int *to_recover, int total_succes_raw, int *survivors){

  int w = 8;

  if(topo->groupRank==0){
    printf("Performing RS decoding...\n");
    fflush(stdout);
  }
  // We first set the matrix which we will need to take the inverse of
  int *matrix,*inverse_matrix;
  if(topo->groupRank==0){
    matrix = (int *)malloc(topo->groupSize*topo->groupSize*sizeof(int));
    int j;
    int X[topo->groupSize], Y[topo->groupSize];
    for(j=0;j<topo->groupSize;j++){
      X[j]=j;
    }
    for(j=topo->groupSize;j<2*topo->groupSize;j++){
      Y[j%topo->groupSize]=j;
    } 
    int i;
    for(i=0;i<topo->groupSize;i++){
      if(survivors[i]<topo->groupSize){
        // The first rows, corresponding to the raw ckpt files, are easy
        for(j=0;j<topo->groupSize;j++){
          matrix[i*topo->groupSize+j]=0;
        }
        matrix[i*topo->groupSize+survivors[i]]=1;
      }else{
        // The rows corresponding to the encoded files are trickier, but are really the same as for the encoding process for the
        // appropiate encoded files.
        for(j=0;j<topo->groupSize;j++){
          matrix[i*topo->groupSize+j]=galois_single_divide(1,(X[survivors[i]%topo->groupSize]^Y[j]),w);
        }
      }
    }
    // Now we should take the inverse of the matrix
    inverse_matrix = (int *)malloc(topo->groupSize*topo->groupSize*sizeof(int));
    jerasure_invert_matrix(matrix,inverse_matrix,topo->groupSize,w);
  }

  // At this point we have to proceed similarly to as we did in the encoding procedure, with the exception that to do the 
  // computations we have to retrieve the survivor files, and we only need to send the resulting recovered files to
  // the processes in to_recover

  // We need to know the ckpt file sizes, which are all the same. So we can get the size from just one process, provided that
  // we are cautious not to select a process which has been corrupted
  off_t ckptFileSize;
  if(survivors[0]<topo->groupSize){
    if(topo->groupRank==survivors[0]){
      int fd_m = open(filename.c_str(), O_RDONLY);
      JASSERT(fd_m != -1);
      struct stat sb;
      JASSERT(fstat(fd_m, &sb) == 0);
      ckptFileSize = sb.st_size;
      close(fd_m);
    }
    MPI_Bcast(&ckptFileSize,sizeof(off_t),MPI_CHAR,survivors[0],topo->groupComm);
  }else if(survivors[0]>=topo->groupSize){
    if(topo->groupRank==survivors[0]%topo->groupSize){
      string filename_encoded = filename + "_encoded";
      int fd_m = open(filename_encoded.c_str(), O_RDONLY);
      JASSERT(fd_m != -1);
      struct stat sb;
      JASSERT(fstat(fd_m, &sb) == 0);
      ckptFileSize = sb.st_size;
      close(fd_m);
    }
    MPI_Bcast(&ckptFileSize,sizeof(off_t),MPI_CHAR,survivors[0]-topo->groupSize,topo->groupComm);
  }

  int blockSize = 1024*1024; 
  
  char *restoredBlocks; // Group rank 0 is going to temporarily store the piece of recovered ckpt images before sending them to
                        // their respective processes
  char *dataBlock; 
  char *restoredBlock; // The to_recover processes will keep getting pieces of their recovered ckpt files

  int num_to_recover = topo->groupSize-total_succes_raw;

  string encodedFilename = filename + "_encoded";
  string encodedChksum = filename + "_encoded_md5chksum";
  int fd_m,fd_e;

  if(topo->groupRank==0){
    restoredBlocks = (char*)malloc(blockSize*num_to_recover);
    dataBlock = (char*)malloc(blockSize);
  }
  int i;
  for(i=0;i<num_to_recover;i++){
    if(topo->groupRank==to_recover[i]){
      restoredBlock =  (char*)malloc(blockSize);
    }
  }
  for(i=0;i<topo->groupSize;i++){
    if(survivors[i]<topo->groupSize){
      if(topo->groupRank==survivors[i]){
        dataBlock = (char*)malloc(blockSize);
        fd_m = open(filename.c_str(), O_RDONLY);
        JASSERT(fd_m != -1);
      }
    }else if (survivors[i]>=topo->groupSize){
      if(topo->groupRank==survivors[i]%topo->groupSize){
        dataBlock = (char*)malloc(blockSize);
        fd_e = open(encodedFilename.c_str(), O_RDONLY);
        JASSERT(fd_e != -1);
      }
    }
  }

  for(i=0;i<num_to_recover;i++){
    if(topo->groupRank==to_recover[i]){
      fd_m = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
      if(truncate(filename.c_str(),0) == -1){
        JASSERT(0).Text("Error with truncation on ckpt image.\n");
      }
      close(fd_m);
      fd_m = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
    }
  }

  int pos = 0;
  while(pos<ckptFileSize){
    if(ckptFileSize-pos<blockSize){
      blockSize=ckptFileSize-pos;
    }
    if(topo->groupRank==0){
      // We have to iteratively acquire the data from survivors[i]
      for(i=0;i<topo->groupSize;i++){
        if(survivors[i]<topo->groupSize){
          if(survivors[i]==0){
            Util::readAll(fd_m,dataBlock,blockSize);
          } else{
            MPI_Recv(dataBlock,blockSize,MPI_CHAR,survivors[i],0,topo->groupComm,MPI_STATUS_IGNORE);
          }
          int j;
          for(j=0;j<num_to_recover;j++){
            int matrix_element = inverse_matrix[to_recover[j]*topo->groupSize+i];
            int k;
            for(k=0;k<blockSize;k++){
              if(i==0){
                restoredBlocks[j*blockSize+k] = galois_single_multiply(matrix_element,dataBlock[k],w);
              }else{
                restoredBlocks[j*blockSize+k] = (restoredBlocks[j*blockSize+k]^galois_single_multiply(matrix_element,dataBlock[k],w));
              }
            }
          }
        }else if(survivors[i]>=topo->groupSize){
          if(survivors[i]%topo->groupSize==0){
            Util::readAll(fd_e,dataBlock,blockSize);
          } else{
            MPI_Recv(dataBlock,blockSize,MPI_CHAR,survivors[i]%topo->groupSize,0,topo->groupComm,MPI_STATUS_IGNORE);
          }
          int j;
          for(j=0;j<num_to_recover;j++){
            int matrix_element = inverse_matrix[to_recover[j]*topo->groupSize+i];
            int k;
            for(k=0;k<blockSize;k++){
              if(i==0){
                restoredBlocks[j*blockSize+k]=galois_single_multiply(matrix_element,dataBlock[k],w);
              }else{
                restoredBlocks[j*blockSize+k] = (restoredBlocks[j*blockSize+k]^galois_single_multiply(matrix_element,dataBlock[k],w));
              }
            }
          }
        }
      }
    } else{
      for(i=0;i<topo->groupSize;i++)
      {
        if(survivors[i]<topo->groupSize && survivors[i]!=0 && topo->groupRank==survivors[i]){
          Util::readAll(fd_m,dataBlock,blockSize);
          MPI_Send(dataBlock,blockSize,MPI_CHAR,0,0,topo->groupComm);
        } else if(survivors[i]>= topo->groupSize && survivors[i]%topo->groupSize!=0 && topo->groupRank==survivors[i]%topo->groupSize){
          Util::readAll(fd_e,dataBlock,blockSize);
          MPI_Send(dataBlock,blockSize,MPI_CHAR,0,0,topo->groupComm);
        }
      }
    }

    // Now we have to send back the pieces of recovered ckpt files to their respective processes, with the caution that 
    // process group Rank 0 can be one of the processes in to_recover and there is no need to send in that.
    for(i=0;i<num_to_recover;i++){
      if(to_recover[i]==0){
        if(topo->groupRank==0){
          Util::writeAll(fd_m,&(restoredBlocks[i*blockSize]),blockSize);
        }
      }else{
        if(topo->groupRank==0){
          MPI_Send(&(restoredBlocks[i*blockSize]),blockSize,MPI_CHAR,to_recover[i],0,topo->groupComm);
        }else{
          if(topo->groupRank==to_recover[i]){
            MPI_Recv(restoredBlock,blockSize,MPI_CHAR,0,0,topo->groupComm,MPI_STATUS_IGNORE);
            Util::writeAll(fd_m,restoredBlock,blockSize);
          }
        }
      }
    }
    MPI_Barrier(topo->groupComm);
    pos +=blockSize;
  }
  
  if(topo->groupRank==0){
    free(inverse_matrix);
    free(restoredBlocks);
    free(dataBlock);
  }
  for(i=0;i<num_to_recover;i++){
    if(topo->groupRank==to_recover[i]){
      free(restoredBlock);
    }
  }

  for(i=0;i<topo->groupSize;i++){
    if(survivors[i]<topo->groupSize){
      if(topo->groupRank==survivors[i] && topo->groupRank!=0){
        free(dataBlock);
      }
    }else if (survivors[i]>=topo->groupSize){
      if(topo->groupRank==survivors[i]%topo->groupSize && topo->groupRank!=0){
        free(dataBlock);
      }
    }
  }

  for(i=0;i<topo->groupSize;i++){
    if(survivors[i]<topo->groupSize){
      if(topo->groupRank==survivors[i]){
        close(fd_m);
      }
    }else if (survivors[i]>=topo->groupSize){
      if(topo->groupRank==survivors[i]%topo->groupSize){
        close(fd_e);
      }
    }
  }

  for(i=0;i<num_to_recover;i++){
    if(topo->groupRank==to_recover[i]){
      close(fd_m);
    }
  }
  
  // Before finishing, we have to restore the ckpt files to their original sizes, which are stored at the end of the still
  // enlarged files
  fd_m = open(filename.c_str(), O_RDONLY);
  JASSERT(fd_m != -1);
  if(lseek(fd_m, -sizeof(off_t), SEEK_END) == -1){
    JASSERT(0).Text("Unable to seek in file.\n");
  }

  off_t original_size;
  if(read(fd_m,&original_size,sizeof(off_t)) == -1){
    JASSERT(0).Text("Unable to read ckpt file size from elongated file.\n");
  }
  printf("Rank %d read ckptFileSize %lu\n",topo->groupRank,original_size);
  fflush(stdout);
  close(fd_m);
  if(truncate(filename.c_str(),original_size) == -1){
    JASSERT(0).Text("Error with truncation ckpt image to its original size.\n");
  }
  MPI_Barrier(topo->groupComm);
}

  
char*
findCkptFilename(string ckpt_dir, string patternEnd){
  int file_found = 0;  
  DIR *dir;
  struct dirent *entry;
  char *result = (char *)malloc(1024);
  dir = opendir(ckpt_dir.c_str());
  if (dir == NULL) {
    printf("  Checkpoint directory %s not found.\n", ckpt_dir.c_str());
    return 0;
  }
  while ((entry = readdir(dir)) != NULL){
    if(Util::strStartsWith(entry->d_name, "ckpt") &&
            Util::strEndsWith(entry->d_name, patternEnd.c_str())){
      sprintf(result, "%s/%s", ckpt_dir.c_str(), entry->d_name);
      file_found = 1;
      break;
    }
  }
  if(!file_found) free(result);

  return (file_found) ? result : NULL;
}


/**
 *
 * Fake reading of ProcessInfo data for accessing later data
 * to avoid modifying the underlying process.
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
  int num_updates = 0;
  MD5_Init(&context);
  char *addr_tmp = (char *)malloc(MEM_TMP_SIZE);
  while (Util::readAll(fd, &area, sizeof(area)) == sizeof(area)){
    //if -1 size found, we are finished
    if (area.size == END_OF_CKPT) break;

    //update context with area info
    MD5_Update(&context, &area, sizeof(area));

    //printf("Reading region of %lu bytes...\n", area.size);
    //fflush(stdout);

    //printf("area name: %s\n", area.name);


    //handle non rwx and anonymous pages
    if (area.prot == 0 || (area.name[0] == '\0' &&
          ((area.flags & MAP_ANONYMOUS) != 0) &&
          ((area.flags & MAP_PRIVATE) != 0))){
      if(area.properties == DMTCP_ZERO_PAGE) {
        //zero page, skip
        continue;
      }
    }

    
    if ((area.properties & DMTCP_SKIP_WRITING_TEXT_SEGMENTS) &&
          (area.prot & PROT_EXEC)){
       //skip text segment if applicable
       continue;
    }
   

    uint64_t toRead = area.size;
    while (toRead > 0){
      uint64_t chunkSize = (toRead > MEM_TMP_SIZE) ?
                           MEM_TMP_SIZE : toRead;

      if((bytes_read = Util::readAll(fd, addr_tmp, chunkSize)) != (ssize_t)chunkSize){
        if(bytes_read == -1) {
          printf("  Error reading memory region: %s\n", strerror(errno));
        }
        else {
          printf("  Expected memory region of %lu bytes, got only %ld.\n", area.size, bytes_read);
        }
        close(fd); close (fd_chksum);
        free(addr_tmp);
        return 0;
      }
      MD5_Update(&context, addr_tmp, chunkSize);
      toRead -= chunkSize;
      num_updates++;
    }

  }

  free(addr_tmp);

  if(area.size != END_OF_CKPT){
    printf("  Checkpoint file did not finish as expected.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  MD5_Final(digest, &context);

  //printf("Number of updates: %d\n", num_updates);
  printf("Rank %d: computed checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", digest[i]);
  }
  printf("\nRank %d: read checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", chksum_read[i]);
  }
  printf("\n");
 
  if(strncmp((char *)digest, (char *)chksum_read, 16) != 0){
    printf("  Rank %d: Computed checksum does not match the checksum file contents.\n", _rank);
    close(fd); close (fd_chksum);
    return 0;
  }

  close(fd);
  close(fd_chksum);
  return 1;
}

int
UtilsMPI::isEncodedCkptValid(const char *filename){
  unsigned char digest[16], chksum_read[16];
  string encodedCkpt = filename;
  string encodedChecksum = encodedCkpt + "_md5chksum";

  int fd_m = open(encodedCkpt.c_str(), O_RDONLY);
  int fd_m_chksum = open(encodedChecksum.c_str(), O_RDONLY);
  
  if (fd_m == -1 || fd_m_chksum == -1){
    //treat the absence or lack of ablity to open
    //either of the two files as a failure
    printf("  Checkpoint or checksum file missing.\n");
    if(fd_m != -1) close(fd_m);
    if(fd_m_chksum != -1) close(fd_m_chksum);
    return 0;
  }

  //read checksum file
  if(Util::readAll(fd_m_chksum, chksum_read, 16) != 16){
    printf("  Checksum file is smaller than 16 bytes.\n");
    close(fd_m); close (fd_m_chksum);
    return 0;
  }

  MD5_CTX context;
  Area area;
  size_t END_OF_CKPT = -1;
  ssize_t bytes_read;
  int chunkSize = MEM_TMP_SIZE;
  char *addr_tmp = (char *)malloc(MEM_TMP_SIZE);
  MD5_Init(&context);


  int pos = 0;
  int cont = true;
  while(cont){
    if((bytes_read = Util::readAll(fd_m, addr_tmp, chunkSize)) != (ssize_t)chunkSize){
        cont = false;
        if(bytes_read == -1) {
          printf("Error reading encoded memory region: %s\n", strerror(errno));
          fflush(stdout);
        }
    }
    MD5_Update(&context,addr_tmp,bytes_read);
    pos += chunkSize;
  }
  MD5_Final(digest, &context);
  close(fd_m);
  close(fd_m_chksum);

  printf("Rank %d: computed encoded checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", digest[i]);
  }
  printf("\nRank %d: read encoded checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", chksum_read[i]);
  }
  printf("\n");
 
  if(strncmp((char *)digest, (char *)chksum_read, 16) != 0){
    printf("  Rank %d: Computed encoded checksum does not match the encoded checksum file contents.\n", _rank);
    close(fd_m); close (fd_m_chksum);
    return 0;
  }

  close(fd_m);
  close(fd_m_chksum);
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
UtilsMPI::checkCkptValid(int ckpt_type, string ckpt_dir, Topology *topo){
  string real_dir = ckpt_dir;
  char *rankchar = (char *)malloc(32);
  sprintf(rankchar, "/ckpt_rank_%d/", _rank);
  real_dir += rankchar;
  int success = 0, allsuccess, partnerSuccess;
  string ckptFilename;
  if(ckpt_type != CKPT_SOLOMON){
    try {
      char *filename = findCkptFilename(real_dir, ".dmtcp");
      if(filename != NULL && isCkptValid(filename)){
          success = 1;
          if(ckpt_type == CKPT_PARTNER){
            //assist partner if necessary
            MPI_Sendrecv(&success, 1, MPI_INT, topo->right, 0, 
              &partnerSuccess, 1, MPI_INT, topo->left, 0, topo->groupComm, MPI_STATUS_IGNORE);
            if(!partnerSuccess){
              //check if our partner ckpt is valid
              printf("Rank %d | Group rank %d: Partner group rank %d requested assist with partner copy.\n", _rank, topo->groupRank, topo->left);
              fflush(stdout);
              int partnerCkptValid = 0;
              char *partnerCkpt = findCkptFilename(real_dir, ".dmtcp_partner");
              if(partnerCkpt != NULL){
                partnerCkptValid = isCkptValid(partnerCkpt);
              }
              if(!partnerCkptValid){
                printf("Rank %d | Group rank %d: Partner copy ckpt not valid.\n", _rank, topo->groupRank);
                fflush(stdout);
              }
              //send help response to partner
              MPI_Send(&partnerCkptValid, 1, MPI_INT, topo->left, 0 ,topo->groupComm);
              if(partnerCkptValid){
                //assist
                if(!assistPartnerCopy(partnerCkpt, topo)){
                  printf("Rank %d | Group rank %d: error while assisting group rank %d with partner copy\n", _rank, topo->groupRank, topo->left);
                  fflush(stdout);
                }
              }
              if(partnerCkpt != NULL) free(partnerCkpt);
            }
          }
      }
      else {
        if(ckpt_type == CKPT_PARTNER){
          //recover from partner if possible
          printf("Rank %d | Group rank %d: Requesting recovery with partner group rank %d\n", _rank, topo->groupRank, topo->right);
          MPI_Sendrecv(&success, 1, MPI_INT, topo->right, 0, 
              &partnerSuccess, 1, MPI_INT, topo->left, 0, topo->groupComm, MPI_STATUS_IGNORE);
          int partnerCkptValid = 0, myCkptValid = 0;
          char *partnerCkpt = NULL;
          if(!partnerSuccess){
            printf("Rank %d | Group rank %d: Partner group rank %d requested assist with partner copy.\n", _rank, topo->groupRank, topo->left);
            fflush(stdout);
            partnerCkpt = findCkptFilename(real_dir, ".dmtcp_partner");
            if(partnerCkpt != NULL){
              partnerCkptValid = isCkptValid(partnerCkpt);
            }
            if(!partnerCkptValid){
              printf("Rank %d | Group rank %d: Partner copy ckpt not valid.\n", _rank, topo->groupRank);
              fflush(stdout);
            }
          }
          //send/recv help response to partner
          MPI_Sendrecv(&partnerCkptValid, 1, MPI_INT, topo->left, 0,
                  &myCkptValid, 1, MPI_INT, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
          if(myCkptValid && (partnerSuccess || partnerCkptValid)){
            int ret;
            string fileCkpt;
            if(filename == NULL){
              //if file not found, construct ckpt file
              filename = (char *)malloc(1024);
              sprintf(filename, "%s/ckptzRestored.dmtcp", real_dir.c_str());
            }
            if(topo->groupRank % 2 == 0 ){
              ret = recoverFromPartnerCopy(filename, topo);
              if(!partnerSuccess){
                assistPartnerCopy(partnerCkpt, topo);
              }
            }
            else {
              if(!partnerSuccess){
                assistPartnerCopy(partnerCkpt, topo);
              }
              ret = recoverFromPartnerCopy(filename, topo);
            }
            //verify ckpt
            if (ret){
              success = isCkptValid(filename);
            }
            if(!ret || !success){
              printf("Rank %d | Group rank %d: Partner ckpt transmission was not successful or ckpt is corrupt.\n", _rank, topo->groupRank);
              fflush(stdout);
            }
          }
          if(partnerCkpt != NULL) free(partnerCkpt);
        }
      }
      if(filename != NULL) free(filename);
    }
    catch (std::exception &e){ //survive unexpected exceptions and assume failure
      printf("Rank %d | Group rank %d: An exception occurred(%s).\n", _rank, topo->groupRank, e.what());
    }
  }else if(ckpt_type==CKPT_SOLOMON){
      try{
        //char *filename = findCkptFilename(real_dir, ".dmtcp");
        //char *filename_encoded = findCkptFilename(real_dir, ".dmtcp");
        string filename = findCkptFilename(real_dir, ".dmtcp");
        string filename_encoded = filename + "_encoded";
        int encoded_success=0;
        if(filename.c_str() != NULL){
          if(isCkptValid(filename.c_str())){
            success=1;
          }
          if(isEncodedCkptValid(filename_encoded.c_str())){
            encoded_success = 1;
          }
        }

        int success_array[topo->groupSize], encoded_success_array[topo->groupSize];
        int total_success_raw=0,total_success_encoded=0;
        MPI_Allgather(&success,1,MPI_INT,success_array,1,MPI_INT,topo->groupComm);
        MPI_Allgather(&encoded_success,1,MPI_INT,encoded_success_array,1,MPI_INT,topo->groupComm);

        // We have to determine whether we can recover from this number of failures 
        int i;
        for(i=0;i<topo->groupSize;i++){
          total_success_raw += success_array[i];
          total_success_encoded += encoded_success_array[i];
        }
        if(total_success_raw + total_success_encoded < topo->groupSize){
          return 0;
        }

        // If all the raw ckpt file images, we can proceed
        if(total_success_raw == topo->groupSize){
          return 1;
        }

        // If we get here, we can recover from the ckpt raw and encoded files. We have to identify which ckpt raw files we need to
        // recover.

        int to_recover[topo->groupSize-total_success_raw];
        int survivors[topo->groupSize]; // We just need topo->groupSize correct files to recover all the ckpt images. We select
                                        // all the unencoded correct ones and, if necesssary, encoded ones.
        int pos_recover = 0, pos_survivor = 0;
        for(i=0;i<topo->groupSize;i++){
          if(success_array[i]==0){
            to_recover[pos_recover]=i;
            pos_recover++;
          }else{
            survivors[pos_survivor]=i;
            pos_survivor++;
          }
        }
        // We add the necessary encoded ckpt files. We will denote them by topo->groupSize + i, where i is the process to which they belong.
        for(i=0;i<topo->groupSize;i++){
          if(pos_survivor==topo->groupSize){
            break;
          }
          if(encoded_success_array[i]==1){
            survivors[pos_survivor]=topo->groupSize+i;
            pos_survivor++;
          }
        }

        // At this point we have already selected the survivor files from which we are going to recover. We would now need to apply
        // RS decoding, via multiplication with the inverse of the adequate matrix. 

        performRSDecoding(filename,topo,to_recover,total_success_raw, survivors);
        return 1;
      }
      catch (std::exception &e){ //survive unexpected exceptions and assume failure
        printf("Rank %d | Group rank %d: An exception occurred(%s).\n", _rank, topo->groupRank, e.what());
      } 
  }

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
    Topology *topo;
    string target = "";
    int i, found = 1, valid = 0, candidate;

    /*
    for(int i = 0; i <= CKPT_GLOBAL; i++){
      printf("type: %d, dir: %s, time: %ld\n", i, restartInfo->ckptDir[i].c_str(), restartInfo->ckptTime[i]);
    }
    fflush(stdout);
    */

    //read restart information
    restartInfo->readRestartInfo();

    //get system topology
    getSystemTopology(cfg, &topo);

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
        if (_rank == 0){
          printf("Checking checkpoint dir %s...\n", target.c_str());
          fflush(stdout);
        }
        valid = checkCkptValid(candidate, target, topo);
        if(!valid){
          //if not valid, set time to zero to invalidate
          //and loop again
          restartInfo->ckptTime[candidate] = 0;
        }
      }
    }

    JASSERT(found).Text("Restart point not found.");

    if(_rank == 0){
      printf("Selected checkpoint type %d with location %s\n", candidate, target.c_str());
      fflush(stdout);
    }

    return target;
}

int 
UtilsMPI::jerasure_invert_matrix(int *mat, int *inv, int rows, int w)
{
  // This functions has been copied from the Fast Galois Field Arithmetic Library in C/C++ by James S. Plank
  int cols, i, j, k, x, rs2;
  int row_start, tmp, inverse;
 
  cols = rows;

  k = 0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      inv[k] = (i == j) ? 1 : 0;
      k++;
    }
  }

  /* First -- convert into upper triangular  */
  for (i = 0; i < cols; i++) {
    row_start = cols*i;

    /* Swap rows if we ave a zero i,i element.  If we can't swap, then the 
       matrix was not invertible  */

    if (mat[row_start+i] == 0) { 
      for (j = i+1; j < rows && mat[cols*j+i] == 0; j++) ;
      if (j == rows) return -1;
      rs2 = j*cols;
      for (k = 0; k < cols; k++) {
        tmp = mat[row_start+k];
        mat[row_start+k] = mat[rs2+k];
        mat[rs2+k] = tmp;
        tmp = inv[row_start+k];
        inv[row_start+k] = inv[rs2+k];
        inv[rs2+k] = tmp;
      }
    }
 
    /* Multiply the row by 1/element i,i  */
    tmp = mat[row_start+i];
    if (tmp != 1) {
      inverse = galois_single_divide(1, tmp, w);
      for (j = 0; j < cols; j++) { 
        mat[row_start+j] = galois_single_multiply(mat[row_start+j], inverse, w);
        inv[row_start+j] = galois_single_multiply(inv[row_start+j], inverse, w);
      }
    }

    /* Now for each j>i, add A_ji*Ai to Aj  */
    k = row_start+i;
    for (j = i+1; j != cols; j++) {
      k += cols;
      if (mat[k] != 0) {
        if (mat[k] == 1) {
          rs2 = cols*j;
          for (x = 0; x < cols; x++) {
            mat[rs2+x] ^= mat[row_start+x];
            inv[rs2+x] ^= inv[row_start+x];
          }
        } else {
          tmp = mat[k];
          rs2 = cols*j;
          for (x = 0; x < cols; x++) {
            mat[rs2+x] ^= galois_single_multiply(tmp, mat[row_start+x], w);
            inv[rs2+x] ^= galois_single_multiply(tmp, inv[row_start+x], w);
          }
        }
      }
    }
  }

  /* Now the matrix is upper triangular.  Start at the top and multiply down  */

  for (i = rows-1; i >= 0; i--) {
    row_start = i*cols;
    for (j = 0; j < i; j++) {
      rs2 = j*cols;
      if (mat[rs2+i] != 0) {
        tmp = mat[rs2+i];
        mat[rs2+i] = 0; 
        for (k = 0; k < cols; k++) {
          inv[rs2+k] ^= galois_single_multiply(tmp, inv[row_start+k], w);
        }
      }
    }
  }
  return 0;
}