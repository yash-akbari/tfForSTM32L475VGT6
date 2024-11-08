/* Linker script for STM32L475VGT6 */
MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 1024K  /* Main Flash memory */
    RAM (xrw)  : ORIGIN = 0x20000000, LENGTH = 96K    /* Main RAM (SRAM1) */
    SRAM2 (xrw) : ORIGIN = 0x10000000, LENGTH = 32K   /* Additional RAM (SRAM2) */
}

SECTIONS
{
    /* Code and read-only data */
    .text : {
        KEEP(*(.isr_vector))        /* Interrupt vector table */
        *(.text*)
        *(.rodata*)
    } > FLASH

    /* Initialized data */
    .data : {
        *(.data*)
    } > RAM AT > FLASH

    /* Uninitialized data */
    .bss : {
        *(.bss*)
    } > RAM

    /* Heap and Stack */
    _end = .;
    _estack = ORIGIN(RAM) + LENGTH(RAM); /* Stack at the end of main RAM */
}
