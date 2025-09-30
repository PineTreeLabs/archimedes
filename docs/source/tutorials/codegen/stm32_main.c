#include <stdbool.h>
#include "iir_filter.h"

// ...

/* USER CODE BEGIN PV */
volatile bool ctrl_flag = false;

// Declare the input, output, and workspace structures
iir_filter_arg_t arg;
iir_filter_res_t res;
iir_filter_work_t work;
/* USER CODE END PV */

// ...

int main(void)
{
    // ...

    /* USER CODE BEGIN Init */
    iir_filter_init(&arg, &res, &work);
    /* USER CODE END Init */

    // ...
    
    /* USER CODE BEGIN WHILE */
    while (1)
    {
        /* USER CODE END WHILE */

        /* USER CODE BEGIN 3 */
        if (ctrl_flag) controller_callback();
    }
    /* USER CODE END 3 */
}

// ...

/* USER CODE BEGIN 4 */
void controller_callback(void) {
    // Same implementation as above
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM3) {  // Or whichever timer you're using
        ctrl_flag = true;
    }
}
/* USER CODE END 4 */

// ...