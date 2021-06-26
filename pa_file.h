#pragma once

#include <crtdefs.h>
#include <vector>
#include <string>

typedef std::vector<std::string> PaVfiles;
typedef unsigned __int64 ui64;

#define PAFileBlockPart	(12 * 1024 * 1024)

enum PaFindFileType
{
	PaFindFileType_File,
	PaFindFileType_Directory
};

/*
    function:     paFileExists
	              �ļ��Ƿ����

        file:     �ļ�·��
      return:     ���ڷ���true�������ڷ���false
*/
bool paFileExists(const char* file);


/*
    function:     paWriteToFile
	              д�����ݵ��ļ�

        file:     ��Ҫд����λ��
        data:     ��Ҫд��������
         len:     ��Ҫд�������ݳ���
      return:     д���ɹ�����true��ʧ��Ϊfalse
*/
bool paWriteToFile(const char* file, const void* data, const size_t len);


/*
    function:     paGetFileSize
	              ��ȡ�ļ��ߴ�

        file:     �ļ�·��
      return:     �����ļ��ߴ�
*/
int paGetFileSize(const char* file);


/*
    function:     paGetFileSize64
	              ��ȡ�ļ��ߴ磬64λ��ʾ

        file:     �ļ�·��
      return:     �����ļ��ߴ�
*/
ui64 paGetFileSize64(const char* file);


/*
    function:     paReadFile
	              ��ȡ�ļ����ݣ���ʹ�õ�ʱ����ʹ��delete�ͷŷ���ֵ��

        file:     �ļ�·��
out_of_file_size: ��ȡ�����ݳ��ȣ�����Ҫ�������ֵ����0
      return:     ���ض�ȡ��������ָ�룬���ڲ�ʹ��new����ġ�
*/
unsigned char* paReadFile(const char* file, size_t* out_of_file_size = 0);


/*
    function:     paReadAt
	              ��ȡ�ļ����ݵ�ָ���Ļ������ڴ�

        file:     �ļ�·��
      buffer:     ���ݴ洢������
len_of_buffer:    �������Ĵ�С
 len_of_read:     �ж������ݱ���ȡ��д�뵽buffer��
      return:     �����ȡ�ɹ��򷵻�true��ʧ�ܷ���false
*/
bool paReadAt(const char* file, void* buffer, size_t len_of_buffer, size_t* len_of_read = 0);


/*
    function:     paFindFiles
	              Ѱ���ļ�

        path:     �ļ�·����·������������\������/,����û��
         out:     �ҵ����ļ�ȫ·�����������ִ�����������ʱ������out�е�����
      filter:     ����������������jpg������*.jpg
inc_sub_dirs:     �Ƿ������Ŀ¼��ָʾ�Ƿ���Ҫ������Ŀ¼�µ������ļ���
   clear_out:     ִ��ǰ�Ƿ������out�����е����ݣ��������Ҫ��Ѱ�ҵõ��Ķ������������뵽out
      return:     �����ҵ����ļ�����
*/
int paFindFiles(const char* path, PaVfiles& out, const char* filter = "*", bool inc_sub_dirs = true, bool clear_out = true, 
	PaFindFileType type = PaFindFileType_File, unsigned int nFilePerDir = 0);


/*
    function:     paFindFiles
	              Ѱ���ļ�

        path:     �ļ�·����·������������\������/,����û��
         out:     �ҵ����ļ�ȫ·�����������ִ�����������ʱ������out�е�����
      filter:     ����������������jpg������*.jpg
inc_sub_dirs:     �Ƿ������Ŀ¼��ָʾ�Ƿ���Ҫ������Ŀ¼�µ������ļ���
   clear_out:     ִ��ǰ�Ƿ������out�����е����ݣ��������Ҫ��Ѱ�ҵõ��Ķ������������뵽out
      return:     �����ҵ����ļ�����
*/
int paFindFilesShort(const char* path, PaVfiles& out, const char* filter = "*", bool inc_sub_dirs = true, bool clear_out = true, 
	PaFindFileType type = PaFindFileType_File, unsigned int nFilePerDir = 0);


/*
    function:     paCompareFile
	              �Ƚ������ļ��Ƿ�һ����

       file1:     �ļ�1
       file2:     �ļ�2
      return:     ������ȫһ�µ�ʱ�򷵻�true�����򷵻�false
*/
bool paCompareFile(const char* file1, const char* file2);


/*
    function:     paCompareFileBig
	              �Ƚ������ļ��Ƿ�һ�������ڴ��ļ��Ƚ�

       file1:     �ļ�1
       file2:     �ļ�2
      return:     ������ȫһ�µ�ʱ�򷵻�true�����򷵻�false
*/
bool paCompareFileBig(const char* file1, const char* file2, size_t cacheSize = PAFileBlockPart);


/*
    function:     paFileName
	              ��ȡ·�����ļ�������׺���ļ���+��׺������3������Ϊ0��ʾ��ȡ��

   full_path:     �ļ���ȫ·���������ǣ���c:/123.abc.txt��
 name_suffix:     �ļ����ͺ�׺������ᱣ�浽������ȥ��ֻ����Ҫ�ṩ�㹻��Ļ���������, ���ｫ���룺��123.abc.txt��
 name_buffer:     �ļ��������������ｫ���룺��123.abc��
suffix_buffer:    ��׺�����������ｫ���룺��txt��
  dir_buffer:     �ļ�Ŀ¼�����������ｫ���룺��c:��
*/
void paFileName(const char* full_path, char* name_suffix = 0, char* name_buffer = 0, char* suffix_buffer = 0, char* dir_buffer = 0);


//dir����û�д�/
void paGetModulePath(char* path = 0, char* dir = 0, char* name_suffix = 0, char* name = 0);


/*
    function:     paChangePathName
	              �޸�·������·������3����ɣ�Ŀ¼+�ļ���+��׺��������������еĲ��֡�����full_pathָ��

   full_path:     �ļ���ȫ·��
	     dir:     ��Ҫ�޸ĵ�Ŀ¼��Ŀ¼������Դ�"/"����"\"���߲���.Ϊ0ʱ����full_path�е�Ŀ¼
        name:     �ļ�����Ϊ0ʱ����full_path�е��ļ���
      suffix:     ��׺��Ϊ0ʱ����full_path�еĺ�׺��Ϊ��ʱ���ļ����ͺ�׺֮��û��"."
*/
char* paChangePathName(char* full_path, const char* dir = 0, const char* name = 0, const char* suffix = 0);


//////////////////////////////////////////////////////////////////////////
//дini�ļ�
bool paWriteIni(const char* app, const char* key, const char* fileName, const char* fmtValue, ...);


//////////////////////////////////////////////////////////////////////////
//��ini�ļ�
bool paReadIni(const char* app, const char* key, const char* fileName, std::string& value);


//////////////////////////////////////////////////////////////////////////
//�����༶Ŀ¼
bool paCreateDirectoryx(const char* dir);