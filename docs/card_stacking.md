In the UNIVAC 36-bit architecture (such as the [UNIVAC 1100/2200 series](https://en.wikipedia.org/wiki/UNIVAC_1100/2200_series)), card stacking and physical hopper/stacker selection are handled as Input/Output (I/O) control functions, rather than direct arithmetic or CPU-internal instructions.

Card handling (including reading, row/column parsing, and post-read or post-punch stacker selection) is controlled by executing an External Function via the I/O channel.

The I/O Mechanism for Card Stacking
-----------------------------------

To issue a physical manipulation command (such as selecting a specific card stacker), the CPU uses the Load Function in Channel (LFC) instruction:

-   Instruction Mnemonic: `LFC` (Sleuth II / Assembly)
-   Octal Opcode: `75 10` [4]

When executing an `LFC` instruction, the computer sends a specific hardware function word down the designated I/O channel assigned to the card reader or punch subsystem.

Function Control Word Formats
-----------------------------

Depending on the specific peripheral controller used (e.g., the integrated card units or subsystem adapter channels), the Function Word loaded via `LFC` dictates the hardware behavior. For a 36-bit system handling raw 80-column binary data or translating to 6-bit Fieldata/9-bit ASCII formats, the control command typically breaks down into the following operational fields:

1.  Stacker Select Command: Directs the card into a specific secondary or error output pocket (e.g., Stacker 1, Stacker 2, or Select Stacker) instead of the normal output bin.
2.  Read/Punch and Feed: Triggers the movement of the next card into the read/punch station while depositing the processed card into the chosen stacker.

Standard Assembly Implementation
--------------------------------

To dynamically control card operations and route cards to a specific stacker, a typical I/O sequence maps out like this:

```
      LFC,s    CH, FUNCWORD   ; 75 10: Load Function in Channel to send
                              ; the stacker/feed directive to the card device.
      JIC      CH, $          ; 75 02: Jump on Input Channel Busy (wait
                              ; for the hardware action to clear).

```

*(Where `CH` represents the assigned card subsystem hardware channel, and `FUNCWORD` points to the memory location containing the binary configuration bits for card-feed combined with stacker selection.)*
