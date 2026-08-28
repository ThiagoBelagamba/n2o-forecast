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

  @Get('conformal-resumo')
  conformalResumo() {
    return this.results.readConformalResumo();
  }

  @Get('setores')
  setores() {
    return this.results.readSetores();
  }

  @Get('cenario-resumo')
  cenarioResumo() {
    return this.results.readCenarioResumo();
  }

  @Get('experimentos')
  experimentos() {
    return this.results.readExperimentos();
  }

  @Get('sintese')
  sintese() {
    return this.results.readSintese();
  }
}
