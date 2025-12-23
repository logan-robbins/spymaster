import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CommandCenterComponent } from './command-center/command-center.component';

@Component({
    selector: 'app-root',
    standalone: true,
    imports: [CommonModule, CommandCenterComponent],
    template: `
    <app-command-center></app-command-center>
  `,
    styles: []
})
export class AppComponent {
    title = 'frontend';
}
