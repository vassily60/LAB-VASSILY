#include <stdio.h>
#include <stdlib.h>

int main(void){
    int n, i;
    double* notes;
    float moyenne;
    printf("Enter the number of notes: ");
    scanf("%d", &n);
    notes = (double*)malloc(n*sizeof(double));
    if (notes == NULL) {
        printf("Memory allocation failed\n");
        exit(0) ;
    }
    for(i=0; i<n; i++){
        printf("Enter your grade %d", i);
        scanf("%lf", &notes[i]);
        moyenne=moyenne+notes[i];
    }
    printf("result: %d", moyenne/n);
    free(notes);
    return 0;
}