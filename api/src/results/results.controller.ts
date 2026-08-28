import { Controller, Get } from '@nestjs/common';
import { ResultsService } from './results.service';

@Controller()
export class ResultsController {
  constructor(private readonly results: ResultsService) {}

  @Get('health')
  health() {
    return { ok: true };
  }

  @Get('charts')
  charts() {
    return this.results.listCharts();
  }

  @Get('metrics')
  metrics() {
    return this.results.readMetrics();
  }

  @Get('comparacao')
  comparacao() {
    return this.results.readComparacao();
  }

  @Get('projecao-resumo')
  projecaoResumo() {
    return this.results.readProjecaoResumo();
  }
}
