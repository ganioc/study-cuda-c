#ifndef __IMGPGM_H__
#define __IMGPGM_H__

#include <stdio.h>

int src_read_pgm(char*name, unsigned char* image, int irows, int icols);
void src_write_pgm(char* name, unsigned char* image, int rows, int cols, char* comment);

int src_read_ppm(char* name, unsigned char* image, int irows, int icols);
int src_write_ppm(char* name, unsigned char* image, int rows, int cols, char* comment);
void get_PgmPpmParams(char*, int*, int*);
void getout_comment(FILE *);
#endif