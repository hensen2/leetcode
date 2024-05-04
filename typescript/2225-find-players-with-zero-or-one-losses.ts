/**
 * 2225 - Find Players With Zero or One Losses
 * Hashmap | Count frequencies
 * Time: O(n*log(n)) | Space: O(n)
 * Time complexity is O(n*log(n)) for sorting
 */

function findWinners(matches: number[][]): number[][] {
  const lossCounts = new Map<number, number>();

  for (const [winner, loser] of matches) {
    lossCounts.set(winner, lossCounts.get(winner) || 0);
    lossCounts.set(loser, (lossCounts.get(loser) || 0) + 1);
  }

  const zeroLoss: number[] = [];
  const oneLoss: number[] = [];

  for (const [player, count] of lossCounts) {
    if (count === 0) zeroLoss.push(player);
    if (count === 1) oneLoss.push(player);
  }

  return [zeroLoss.sort((a, b) => a - b), oneLoss.sort((a, b) => a - b)];
}
