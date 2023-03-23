from transformers import pipeline
from model_pipelines.layoutlm_document_qa import LayoutlmDocumentQa
import io
import PIL.Image as Image
from array import array

p = pipeline("text-classification",
             model=r"C:\Users\Itay Gafni\Projects\Siri\models\text\Vulnerability-detection",
             return_all_scores=True)
p._postprocess_params['top_k'] = 10
res = p("""
// A C program to demonstrate buffer overflow
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
int main(int argc, char *argv[])
{
 
       // Reserve 5 byte of buffer plus the terminating NULL.
       // should allocate 8 bytes = 2 double words,
       // To overflow, need more than 8 bytes...
       char buffer[5];  // If more than 8 characters input
                        // by user, there will be access
                        // violation, segmentation fault
 
       // a prompt how to execute the program...
       if (argc < 2)
       {
              printf("strcpy() NOT executed....\n");
              printf("Syntax: %s <characters>\n", argv[0]);
              exit(0);
       }
 
       // copy the user input to mybuffer, without any
       // bound checking a secure version is strcpy_s()
       strcpy(buffer, argv[1]);
       printf("buffer content= %s\n", buffer);
 
       // you may want to try strcpy_s()
       printf("strcpy() executed...\n");
 
       return 0;
}"""
)

print(res)
stop = ""

